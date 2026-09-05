use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use flybrain_engine::cns_pathway::{CnsPathway, PathwayControl};
use flybrain_engine::fixture::TickFixture;
use flybrain_engine::metal_engine::{MetalEngine, MetalRun};
use flybrain_engine::output::{NativeRunMetadata, write_native_run};
use flybrain_engine::pack::ConnectomePack;
use flybrain_engine::parameters::ModelParameters;
use flybrain_engine::protocol::sugar_indices;
use flybrain_engine::reference::CpuEngine;
use flybrain_engine::stimulus::EventSchedule;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(name = "flybrain-rs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Audit {
        #[arg(long)]
        pack: PathBuf,
    },
    VerifyFixture {
        #[arg(long, default_value = "fixtures/tiny-parity-v1.json")]
        fixture: PathBuf,
    },
    VerifyEngine {
        #[arg(long)]
        pack: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 150.0)]
        rate_hz: f64,
        #[arg(long, default_value_t = 20260816)]
        seed: u64,
        #[arg(long, default_value_t = 256)]
        chunk_steps: usize,
    },
    Simulate {
        #[arg(long)]
        pack: PathBuf,
        #[arg(long, default_value_t = 10000)]
        steps: usize,
        #[arg(long, default_value_t = 150.0)]
        rate_hz: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 256)]
        chunk_steps: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Pathway(PathwayOptions),
}

#[derive(Debug, clap::Args)]
struct PathwayOptions {
    #[arg(long)]
    pack: PathBuf,
    #[arg(long)]
    pathway: PathBuf,
    #[arg(long, value_enum)]
    control: PathwayControl,
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    #[arg(long, default_value_t = 150.0)]
    rate_hz: f64,
    #[arg(long, default_value_t = 20260816)]
    seed: u64,
    #[arg(long, default_value_t = 256)]
    chunk_steps: usize,
    #[arg(long)]
    verify_cpu: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Audit { pack } => audit(pack),
        Command::VerifyFixture { fixture } => verify_fixture(fixture),
        Command::VerifyEngine {
            pack,
            steps,
            rate_hz,
            seed,
            chunk_steps,
        } => verify_engine(pack, steps, rate_hz, seed, chunk_steps),
        Command::Simulate {
            pack,
            steps,
            rate_hz,
            seed,
            chunk_steps,
            output,
        } => simulate(pack, steps, rate_hz, seed, chunk_steps, output),
        Command::Pathway(options) => pathway_assay(options),
    }
}

fn audit(path: PathBuf) -> Result<()> {
    let pack = ConnectomePack::open(path)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "materialization": pack.materialization(),
            "neuron_count": pack.neuron_count(),
            "edge_count": pack.edge_count(),
            "contact_sum": pack.manifest.contact_sum,
            "array_hashes_verified": true,
        }))?
    );
    Ok(())
}

fn verify_fixture(path: PathBuf) -> Result<()> {
    let fixture = TickFixture::load(path)?;
    let connectome = fixture.connectome()?;
    let schedule = fixture.event_schedule()?;
    let mut cpu = CpuEngine::new(
        &connectome,
        fixture.parameters,
        Some(&fixture.initial_state.v_mv),
        Some(&fixture.initial_state.g_mv),
        &fixture.overrides.zero_refractory,
        &fixture.overrides.silenced_sources,
    )?;
    let cpu_trace = cpu.run_schedule(&schedule, true)?;
    let mut metal = MetalEngine::new(
        &connectome,
        fixture.parameters,
        Some(&fixture.initial_state.v_mv),
        Some(&fixture.initial_state.g_mv),
        &fixture.overrides.zero_refractory,
        &fixture.overrides.silenced_sources,
    )?;
    let metal_trace = metal.run_recorded(&schedule)?;

    let expected_events: Vec<_> = fixture
        .expected
        .spike_events
        .iter()
        .map(|event| (event.tick, event.neuron))
        .collect();
    let cpu_events = spike_events(
        &cpu_trace
            .iter()
            .map(|state| &state.spikes)
            .collect::<Vec<_>>(),
    );
    let metal_events = spike_events(
        &metal_trace
            .iter()
            .map(|state| &state.spikes)
            .collect::<Vec<_>>(),
    );
    let mut cpu_v_error = 0.0_f64;
    let mut cpu_g_error = 0.0_f64;
    let mut metal_v_error = 0.0_f64;
    let mut metal_g_error = 0.0_f64;
    for tick in 0..fixture.run.steps {
        for neuron in 0..connectome.neuron_count() {
            cpu_v_error = cpu_v_error.max(
                (cpu_trace[tick].voltage_mv[neuron] - fixture.expected.v_end_mv[tick][neuron])
                    .abs(),
            );
            cpu_g_error = cpu_g_error.max(
                (cpu_trace[tick].conductance_mv[neuron] - fixture.expected.g_end_mv[tick][neuron])
                    .abs(),
            );
            metal_v_error = metal_v_error.max(
                (metal_trace[tick].voltage_mv[neuron] as f64
                    - fixture.expected.v_end_mv[tick][neuron])
                    .abs(),
            );
            metal_g_error = metal_g_error.max(
                (metal_trace[tick].conductance_mv[neuron] as f64
                    - fixture.expected.g_end_mv[tick][neuron])
                    .abs(),
            );
        }
    }
    let cpu_spikes_exact = cpu_events == expected_events;
    let metal_spikes_exact = metal_events == expected_events;
    if !cpu_spikes_exact || !metal_spikes_exact {
        bail!("fixture spike events do not match");
    }
    if cpu_v_error > fixture.acceptance.state_abs_tol_mv
        || cpu_g_error > fixture.acceptance.state_abs_tol_mv
    {
        bail!("Rust float64 state exceeds the fixture tolerance");
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "case_id": fixture.case_id,
            "steps": fixture.run.steps,
            "cpu_spikes_exact": cpu_spikes_exact,
            "metal_spikes_exact": metal_spikes_exact,
            "cpu_max_voltage_error_mv": cpu_v_error,
            "cpu_max_conductance_error_mv": cpu_g_error,
            "metal_max_voltage_error_mv": metal_v_error,
            "metal_max_conductance_error_mv": metal_g_error,
            "metal_device": metal.device_name(),
        }))?
    );
    Ok(())
}

fn verify_engine(
    path: PathBuf,
    steps: usize,
    rate_hz: f64,
    seed: u64,
    chunk_steps: usize,
) -> Result<()> {
    if steps == 0 {
        bail!("steps must be positive");
    }
    let pack = ConnectomePack::open(path)?;
    let parameters = ModelParameters::default();
    let (targets, missing) = sugar_indices(&pack);
    if targets.is_empty() {
        bail!("none of the sugar-neuron IDs exist in this pack");
    }
    let schedule = EventSchedule::bernoulli(
        targets.clone(),
        steps,
        rate_hz,
        parameters.dt_ms,
        seed,
        pack.neuron_count(),
    )?;

    let mut cpu = CpuEngine::new(&pack, parameters, None, None, &targets, &[])?;
    let cpu_started = Instant::now();
    cpu.run_schedule(&schedule, false)?;
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    let mut metal = MetalEngine::new(&pack, parameters, None, None, &targets, &[])?;
    let result = metal.run_schedule(&schedule, chunk_steps)?;
    let mismatched_indices: Vec<_> = cpu
        .spike_counts()
        .iter()
        .zip(&result.spike_counts)
        .enumerate()
        .filter_map(|(index, (expected, actual))| (expected != actual).then_some(index))
        .collect();
    let max_v_error = cpu
        .voltage_mv()
        .iter()
        .zip(&result.voltage_mv)
        .map(|(expected, actual)| (expected - *actual as f64).abs())
        .fold(0.0_f64, f64::max);
    let max_g_error = cpu
        .conductance_mv()
        .iter()
        .zip(&result.conductance_mv)
        .map(|(expected, actual)| (expected - *actual as f64).abs())
        .fold(0.0_f64, f64::max);
    let mismatched_ids: Vec<_> = mismatched_indices
        .iter()
        .take(20)
        .map(|&index| pack.neuron_ids[index])
        .collect();
    let biological_seconds = steps as f64 * parameters.dt_ms / 1000.0;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "materialization": pack.materialization(),
            "steps": steps,
            "biological_ms": steps as f64 * parameters.dt_ms,
            "stimulus_rate_hz": rate_hz,
            "seed": seed,
            "stimulus_target_count": targets.len(),
            "missing_stimulus_ids": missing,
            "cpu_seconds": cpu_seconds,
            "metal_seconds": result.elapsed.as_secs_f64(),
            "metal_realtime_factor": result.elapsed.as_secs_f64() / biological_seconds,
            "speedup": cpu_seconds / result.elapsed.as_secs_f64(),
            "cpu_spikes": cpu.spike_counts().iter().map(|value| u64::from(*value)).sum::<u64>(),
            "metal_spikes": result.spike_counts.iter().map(|value| u64::from(*value)).sum::<u64>(),
            "metal_spike_counts_sha256": sha256_u32(&result.spike_counts),
            "metal_voltage_sha256": sha256_f32(&result.voltage_mv),
            "metal_conductance_sha256": sha256_f32(&result.conductance_mv),
            "per_neuron_spike_counts_exact": mismatched_indices.is_empty(),
            "mismatched_spike_count_neurons": mismatched_indices.len(),
            "first_mismatched_neuron_ids": mismatched_ids,
            "maximum_voltage_error_mv": max_v_error,
            "maximum_conductance_error_mv": max_g_error,
            "metal_device": metal.device_name(),
            "metal_allocated_bytes": metal.allocated_bytes(),
            "chunk_steps": chunk_steps,
        }))?
    );
    Ok(())
}

fn simulate(
    path: PathBuf,
    steps: usize,
    rate_hz: f64,
    seed: u64,
    chunk_steps: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    if steps == 0 {
        bail!("steps must be positive");
    }
    let pack = ConnectomePack::open(path)?;
    let parameters = ModelParameters::default();
    let (targets, missing) = sugar_indices(&pack);
    if targets.is_empty() {
        bail!("none of the sugar-neuron IDs exist in this pack");
    }
    let schedule = EventSchedule::bernoulli(
        targets.clone(),
        steps,
        rate_hz,
        parameters.dt_ms,
        seed,
        pack.neuron_count(),
    )?;
    let mut engine = MetalEngine::new(&pack, parameters, None, None, &targets, &[])?;
    let result = engine.run_schedule(&schedule, chunk_steps)?;
    let biological_seconds = steps as f64 * parameters.dt_ms / 1000.0;
    let total_spikes = result
        .spike_counts
        .iter()
        .map(|value| u64::from(*value))
        .sum::<u64>();
    if let Some(output) = output {
        let manifest = write_native_run(
            output,
            &pack,
            &result,
            NativeRunMetadata {
                parameters,
                steps,
                rate_hz,
                seed,
                stimulus_target_count: targets.len(),
                missing_stimulus_ids: &missing,
                chunk_steps,
                device_name: engine.device_name(),
                allocated_bytes: engine.allocated_bytes(),
            },
        )?;
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "materialization": pack.materialization(),
            "steps": steps,
            "biological_ms": biological_seconds * 1000.0,
            "stimulus": "right_sugar_grns",
            "stimulus_generator": "splitmix64-counter-v1",
            "stimulus_rate_hz": rate_hz,
            "stimulus_target_count": targets.len(),
            "missing_stimulus_ids": missing,
            "seed": seed,
            "total_spikes": total_spikes,
            "active_neurons": result.spike_counts.iter().filter(|value| **value != 0).count(),
            "spike_counts_sha256": sha256_u32(&result.spike_counts),
            "voltage_sha256": sha256_f32(&result.voltage_mv),
            "conductance_sha256": sha256_f32(&result.conductance_mv),
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "realtime_factor": result.elapsed.as_secs_f64() / biological_seconds,
            "metal_device": engine.device_name(),
            "metal_allocated_bytes": engine.allocated_bytes(),
            "chunk_steps": chunk_steps,
        }))?
    );
    Ok(())
}

fn pathway_assay(options: PathwayOptions) -> Result<()> {
    let PathwayOptions {
        pack: pack_path,
        pathway: pathway_path,
        control,
        steps,
        rate_hz,
        seed,
        chunk_steps,
        verify_cpu,
        output,
    } = options;
    let parameters = ModelParameters::default().validate()?;
    validate_pathway_options(steps, rate_hz, chunk_steps, parameters, output.as_deref())?;

    let pathway_bytes = fs::read(&pathway_path)
        .with_context(|| format!("reading pathway artifact {}", pathway_path.display()))?;
    let pathway_artifact_sha256 = sha256_bytes(&pathway_bytes);
    let pathway = CnsPathway::load(&pathway_path)?;
    let pathway_fingerprint_sha256 = pathway.fingerprint()?;
    let pack = ConnectomePack::open(&pack_path)?;
    let resolved = pathway.resolve(&pack)?;
    let (driven_indices, driven_role) = if control == PathwayControl::RelayDriven {
        (resolved.relay_indices.as_slice(), "relay")
    } else {
        (resolved.stimulus_indices.as_slice(), "input")
    };
    let silenced_sources = resolved.silenced_sources(control);
    if steps.checked_mul(driven_indices.len()).is_none() {
        bail!("stimulus dimensions overflow");
    }

    let schedule = resolved.schedule(
        control,
        steps,
        rate_hz,
        seed,
        parameters,
        pack.neuron_count(),
    )?;
    let stimulus_event_count = schedule
        .counts()
        .iter()
        .filter(|&&count| count != 0)
        .count();
    let biological_seconds = steps as f64 * parameters.dt_ms / 1000.0;
    let zero_refractory = driven_indices;
    let driven_ids: Vec<u64> = driven_indices
        .iter()
        .map(|&index| pack.neuron_ids[index as usize])
        .collect();
    let mut metal = MetalEngine::new(
        &pack,
        parameters,
        None,
        None,
        zero_refractory,
        silenced_sources,
    )?;
    let result = metal.run_schedule(&schedule, chunk_steps)?;

    let cpu_verification = if verify_cpu {
        Some(verify_pathway_cpu(
            &pack,
            parameters,
            &schedule,
            zero_refractory,
            silenced_sources,
            &result,
        )?)
    } else {
        None
    };

    let input_metrics = group_metrics(
        &resolved.stimulus_indices,
        &pack.neuron_ids,
        &result.spike_counts,
        biological_seconds,
    );
    let relay_metrics = group_metrics(
        &resolved.relay_indices,
        &pack.neuron_ids,
        &result.spike_counts,
        biological_seconds,
    );
    let readout_metrics: serde_json::Map<String, serde_json::Value> = resolved
        .readout_indices
        .iter()
        .map(|(name, indices)| {
            (
                name.clone(),
                group_metrics(
                    indices,
                    &pack.neuron_ids,
                    &result.spike_counts,
                    biological_seconds,
                ),
            )
        })
        .collect();
    let total_population = population_metrics(&result.spike_counts, biological_seconds);
    let pack_manifest_sha256 = pack
        .path
        .as_ref()
        .map(|root| root.join("manifest.json"))
        .map(|path| {
            fs::read(&path)
                .with_context(|| format!("reading pack manifest {}", path.display()))
                .map(|bytes| sha256_bytes(&bytes))
        })
        .transpose()?;
    let state_sha256 = sha256_state_f32(&result.voltage_mv, &result.conductance_mv);
    let report = json!({
        "schema": "flybrain.cns-pathway-assay",
        "schema_version": 1,
        "assay": "native_cns_pathway",
        "engine": "rust-objc2-metal",
        "pack": {
            "path": pack_path,
            "materialization": pack.materialization(),
            "neuron_count": pack.neuron_count(),
            "edge_count": pack.edge_count(),
            "contact_sum": pack.manifest.contact_sum,
            "manifest_sha256": pack_manifest_sha256,
            "source_sha256": &pack.manifest.source_sha256,
            "array_sha256": &pack.manifest.array_sha256,
        },
        "pathway": {
            "path": pathway_path,
            "name": pathway.name,
            "materialization": pathway.materialization,
            "fingerprint_sha256": pathway_fingerprint_sha256,
            "artifact_sha256": pathway_artifact_sha256,
            "evidence": pathway.evidence,
            "interpretation": pathway.interpretation,
            "anatomical_edge_count": resolved.anatomical_edge_count,
        },
        "control": pathway_control_name(control),
        "driven_role": driven_role,
        "model_parameters": parameters,
        "parameter_source": "ModelParameters::default (Shiu defaults)",
        "cns_physiology_validation": "unvalidated",
        "interpretation_note": "The default Shiu parameters are an explicit computational configuration; this assay does not validate CNS physiology.",
        "body_behavior": "not_simulated",
        "steps": steps,
        "dt_ms": parameters.dt_ms,
        "duration_ms": steps as f64 * parameters.dt_ms,
        "seed": seed,
        "chunk_steps": chunk_steps,
        "stimulus": {
            "protocol": "Bernoulli external activation proxy",
            "generator": "splitmix64-counter-v1",
            "driven_role": driven_role,
            "rate_hz": rate_hz,
            "event_count": stimulus_event_count,
            "target_count": driven_indices.len(),
            "target_ids": &driven_ids,
            "scheduled_target_count": schedule.targets().len(),
            "zero_refractory_neuron_count": zero_refractory.len(),
            "zero_refractory_neuron_ids": &driven_ids,
        },
        "silencing": {
            "source_count": silenced_sources.len(),
            "source_ids": silenced_sources
                .iter()
                .map(|&index| pack.neuron_ids[index as usize])
                .collect::<Vec<_>>(),
        },
        "inputs": input_metrics,
        "relays": relay_metrics,
        "readouts": readout_metrics,
        "population": total_population,
        "full_hashes": {
            "spike_counts_sha256": sha256_u32(&result.spike_counts),
            "voltage_final_mv_sha256": sha256_f32(&result.voltage_mv),
            "conductance_final_mv_sha256": sha256_f32(&result.conductance_mv),
            "state_sha256": state_sha256,
        },
        "timing": {
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "biological_seconds": biological_seconds,
            "realtime_factor": result.elapsed.as_secs_f64() / biological_seconds,
            "device_name": metal.device_name(),
            "allocated_bytes": metal.allocated_bytes(),
        },
        "cpu_verification": cpu_verification,
    });
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(output) = output {
        write_json_create_new(&output, &serialized)?;
    }
    println!("{serialized}");
    Ok(())
}

fn validate_pathway_options(
    steps: usize,
    rate_hz: f64,
    chunk_steps: usize,
    parameters: ModelParameters,
    output: Option<&Path>,
) -> Result<()> {
    if steps == 0 {
        bail!("steps must be positive");
    }
    if !rate_hz.is_finite() || rate_hz < 0.0 {
        bail!("stimulus rate must be finite and non-negative");
    }
    if rate_hz * parameters.dt_ms / 1000.0 > 1.0 {
        bail!("rate × timestep cannot exceed one for N=1 input");
    }
    if chunk_steps == 0 {
        bail!("chunk_steps must be positive");
    }
    if let Some(output) = output {
        validate_json_output_path(output)?;
    }
    Ok(())
}

fn validate_json_output_path(output: &Path) -> Result<()> {
    if output.as_os_str().is_empty() || output.file_name().is_none() {
        bail!("output must be a JSON file path");
    }
    if output.exists() || output.is_symlink() {
        bail!("output file already exists: {}", output.display());
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!("output parent is not a directory: {}", parent.display());
    }
    Ok(())
}

fn write_json_create_new(output: &Path, serialized: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| format!("creating output {}", output.display()))?;
    let write_result = file
        .write_all(serialized.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.flush());
    if let Err(error) = write_result {
        drop(file);
        let _ = fs::remove_file(output);
        return Err(error.into());
    }
    Ok(())
}

fn verify_pathway_cpu(
    pack: &ConnectomePack,
    parameters: ModelParameters,
    schedule: &EventSchedule,
    zero_refractory: &[u32],
    silenced_sources: &[u32],
    metal: &MetalRun,
) -> Result<serde_json::Value> {
    let mut cpu = CpuEngine::new(
        pack,
        parameters,
        None,
        None,
        zero_refractory,
        silenced_sources,
    )?;
    cpu.run_schedule(schedule, false)?;
    let mismatched_count_neurons = cpu
        .spike_counts()
        .iter()
        .zip(&metal.spike_counts)
        .filter(|(cpu, metal)| *cpu != *metal)
        .count();
    let mismatched_count_total = cpu
        .spike_counts()
        .iter()
        .zip(&metal.spike_counts)
        .map(|(cpu, metal)| (i64::from(*cpu) - i64::from(*metal)).unsigned_abs())
        .sum::<u64>();
    let max_voltage_error_mv = cpu
        .voltage_mv()
        .iter()
        .zip(&metal.voltage_mv)
        .map(|(cpu, metal)| (*cpu - f64::from(*metal)).abs())
        .fold(0.0_f64, f64::max);
    let max_conductance_error_mv = cpu
        .conductance_mv()
        .iter()
        .zip(&metal.conductance_mv)
        .map(|(cpu, metal)| (*cpu - f64::from(*metal)).abs())
        .fold(0.0_f64, f64::max);
    Ok(json!({
        "spike_counts_exact": mismatched_count_neurons == 0,
        "mismatched_count_neurons": mismatched_count_neurons,
        "mismatched_count_total": mismatched_count_total,
        "maximum_voltage_error_mv": max_voltage_error_mv,
        "maximum_conductance_error_mv": max_conductance_error_mv,
    }))
}

fn group_metrics(
    indices: &[u32],
    neuron_ids: &[u64],
    spike_counts: &[u32],
    biological_seconds: f64,
) -> serde_json::Value {
    let mut neurons = Vec::with_capacity(indices.len());
    let mut counts = Vec::with_capacity(indices.len());
    let mut mean_rates_hz = Vec::with_capacity(indices.len());
    let mut total_spikes = 0_u64;
    let mut active_neurons = 0_usize;
    for &index in indices {
        let count = spike_counts[index as usize];
        let mean_rate_hz = f64::from(count) / biological_seconds;
        total_spikes += u64::from(count);
        active_neurons += if count != 0 { 1 } else { 0 };
        counts.push(count);
        mean_rates_hz.push(mean_rate_hz);
        neurons.push(json!({
            "neuron_id": neuron_ids[index as usize],
            "index": index,
            "spike_count": count,
            "mean_rate_hz": mean_rate_hz,
        }));
    }
    let mean_rate_hz = if indices.is_empty() {
        0.0
    } else {
        total_spikes as f64 / biological_seconds / indices.len() as f64
    };
    json!({
        "neuron_count": indices.len(),
        "neurons": neurons,
        "spike_counts": counts,
        "mean_rates_hz": mean_rates_hz,
        "total_spikes": total_spikes,
        "active_neurons": active_neurons,
        "mean_rate_hz": mean_rate_hz,
    })
}

fn population_metrics(spike_counts: &[u32], biological_seconds: f64) -> serde_json::Value {
    let mut total_spikes = 0_u64;
    let mut active_neurons = 0_usize;
    for &count in spike_counts {
        total_spikes += u64::from(count);
        active_neurons += if count != 0 { 1 } else { 0 };
    }
    let mean_rate_hz = if spike_counts.is_empty() {
        0.0
    } else {
        total_spikes as f64 / biological_seconds / spike_counts.len() as f64
    };
    json!({
        "neuron_count": spike_counts.len(),
        "total_spikes": total_spikes,
        "active_neurons": active_neurons,
        "mean_rate_hz": mean_rate_hz,
    })
}

fn pathway_control_name(control: PathwayControl) -> &'static str {
    match control {
        PathwayControl::Intact => "intact",
        PathwayControl::NoInput => "no-input",
        PathwayControl::InputDisconnected => "input-disconnected",
        PathwayControl::RelayDisconnected => "relay-disconnected",
        PathwayControl::RelayDriven => "relay-driven",
    }
}

fn spike_events(spikes: &[&Vec<u8>]) -> Vec<(usize, usize)> {
    let mut events = Vec::new();
    for (tick, row) in spikes.iter().enumerate() {
        for (neuron, fired) in row.iter().enumerate() {
            if *fired != 0 {
                events.push((tick, neuron));
            }
        }
    }
    events
}

fn sha256_u32(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn sha256_state_f32(voltage_mv: &[f32], conductance_mv: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in voltage_mv {
        digest.update(value.to_bits().to_le_bytes());
    }
    for value in conductance_mv {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_f32(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::{ModelParameters, Path, validate_pathway_options};

    #[test]
    fn pathway_options_reject_zero_steps() {
        assert!(validate_pathway_options(0, 150.0, 256, ModelParameters::default(), None).is_err());
    }

    #[test]
    fn pathway_options_reject_zero_chunk_steps() {
        assert!(validate_pathway_options(1, 150.0, 0, ModelParameters::default(), None).is_err());
    }

    #[test]
    fn pathway_options_reject_non_finite_rate() {
        assert!(
            validate_pathway_options(1, f64::NAN, 256, ModelParameters::default(), None).is_err()
        );
    }

    #[test]
    fn pathway_options_reject_invalid_output_path() {
        assert!(
            validate_pathway_options(
                1,
                150.0,
                256,
                ModelParameters::default(),
                Some(Path::new("/")),
            )
            .is_err()
        );
    }
}
