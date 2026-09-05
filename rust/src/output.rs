use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::metal_engine::MetalRun;
use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;

pub struct NativeRunMetadata<'a> {
    pub parameters: ModelParameters,
    pub steps: usize,
    pub rate_hz: f64,
    pub seed: u64,
    pub stimulus_target_count: usize,
    pub missing_stimulus_ids: &'a [u64],
    pub chunk_steps: usize,
    pub device_name: &'a str,
    pub allocated_bytes: usize,
}

pub fn write_native_run(
    output_path: impl AsRef<Path>,
    connectome: &ConnectomePack,
    run: &MetalRun,
    metadata: NativeRunMetadata<'_>,
) -> Result<Value> {
    let output = output_path.as_ref();
    metadata.parameters.validate()?;
    if metadata.steps == 0 {
        bail!("native run must contain at least one step");
    }
    if !metadata.rate_hz.is_finite() || metadata.rate_hz < 0.0 {
        bail!("stimulus rate must be finite and non-negative");
    }
    if metadata.chunk_steps == 0 {
        bail!("chunk_steps must be positive");
    }
    let neuron_count = connectome.neuron_count();
    if run.spike_counts.len() != neuron_count
        || run.voltage_mv.len() != neuron_count
        || run.conductance_mv.len() != neuron_count
    {
        bail!("native result arrays must have one value per neuron");
    }
    if output.exists() || output.is_symlink() {
        bail!("output directory already exists: {}", output.display());
    }
    let parent = output
        .parent()
        .context("output directory must have a parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating output parent {}", parent.display()))?;
    let name = output
        .file_name()
        .context("output directory must have a file name")?
        .to_string_lossy();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&temporary)
        .with_context(|| format!("creating temporary output {}", temporary.display()))?;

    let result = (|| -> Result<Value> {
        let spike_counts: Vec<i32> = run
            .spike_counts
            .iter()
            .map(|&count| i32::try_from(count).context("spike count exceeds int32"))
            .collect::<Result<_>>()?;
        let biological_seconds = metadata.steps as f64 * metadata.parameters.dt_ms / 1000.0;
        let firing_rates: Vec<f64> = run
            .spike_counts
            .iter()
            .map(|&count| count as f64 / biological_seconds)
            .collect();
        write_npy(&temporary.join("neuron_ids.npy"), &connectome.neuron_ids)?;
        write_npy(&temporary.join("spike_counts.npy"), &spike_counts)?;
        write_npy(&temporary.join("firing_rates_hz.npy"), &firing_rates)?;
        write_npy(&temporary.join("voltage_final_mv.npy"), &run.voltage_mv)?;
        write_npy(
            &temporary.join("conductance_final_mv.npy"),
            &run.conductance_mv,
        )?;
        let output_hashes = [
            "neuron_ids.npy",
            "spike_counts.npy",
            "firing_rates_hz.npy",
            "voltage_final_mv.npy",
            "conductance_final_mv.npy",
        ]
        .into_iter()
        .map(|name| {
            Ok((
                name.to_owned(),
                Value::String(sha256_file(&temporary.join(name))?),
            ))
        })
        .collect::<Result<serde_json::Map<String, Value>>>()?;
        let total_spikes = run
            .spike_counts
            .iter()
            .map(|&value| u64::from(value))
            .sum::<u64>();
        let manifest = json!({
            "schema_version": 1,
            "engine": "rust-objc2-metal",
            "materialization": connectome.materialization(),
            "steps": metadata.steps,
            "dt_ms": metadata.parameters.dt_ms,
            "duration_ms": metadata.steps as f64 * metadata.parameters.dt_ms,
            "model_parameters": metadata.parameters,
            "stimulus": "right_sugar_grns",
            "stimulus_generator": "splitmix64-counter-v1",
            "stimulus_rate_hz": metadata.rate_hz,
            "stimulus_target_count": metadata.stimulus_target_count,
            "missing_stimulus_ids": metadata.missing_stimulus_ids,
            "seed": metadata.seed,
            "chunk_steps": metadata.chunk_steps,
            "metal_device": metadata.device_name,
            "metal_allocated_bytes": metadata.allocated_bytes,
            "elapsed_seconds": run.elapsed.as_secs_f64(),
            "realtime_factor": run.elapsed.as_secs_f64() / biological_seconds,
            "total_spikes": total_spikes,
            "active_neurons": run.spike_counts.iter().filter(|&&count| count != 0).count(),
            "source_array_sha256": &connectome.manifest.array_sha256,
            "output_array_sha256": output_hashes,
        });
        let mut manifest_file = BufWriter::new(File::create(temporary.join("manifest.json"))?);
        serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
        manifest_file.write_all(b"\n")?;
        manifest_file.flush()?;
        Ok(manifest)
    })();

    match result {
        Ok(manifest) => {
            fs::rename(&temporary, output).with_context(|| {
                format!(
                    "committing temporary output {} to {}",
                    temporary.display(),
                    output.display()
                )
            })?;
            Ok(manifest)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            Err(error)
        }
    }
}

trait NpyElement: Copy {
    const DESCR: &'static str;
    fn write_le(self, writer: &mut impl Write) -> std::io::Result<()>;
}

impl NpyElement for u64 {
    const DESCR: &'static str = "<u8";

    fn write_le(self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl NpyElement for i32 {
    const DESCR: &'static str = "<i4";

    fn write_le(self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }
}

impl NpyElement for f64 {
    const DESCR: &'static str = "<f8";

    fn write_le(self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.to_bits().to_le_bytes())
    }
}

impl NpyElement for f32 {
    const DESCR: &'static str = "<f4";

    fn write_le(self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.to_bits().to_le_bytes())
    }
}

fn write_npy<T: NpyElement>(path: &Path, values: &[T]) -> Result<()> {
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({},), }}",
        T::DESCR,
        values.len()
    )
    .into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let header_length = u16::try_from(header.len()).context("NPY header exceeds v1 limit")?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;
    writer.write_all(&header_length.to_le_bytes())?;
    writer.write_all(&header)?;
    for &value in values {
        value.write_le(&mut writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::{NativeRunMetadata, write_native_run};
    use crate::metal_engine::MetalRun;
    use crate::npy;
    use crate::pack::ConnectomePack;
    use crate::parameters::ModelParameters;
    use std::fs;
    use std::time::Duration;

    #[test]
    fn writes_atomic_hashed_native_result() {
        let root = std::env::temp_dir().join(format!(
            "flybrain-output-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let output = root.join("run");
        let connectome = ConnectomePack::from_arrays([10_u64, 20], [0_u32, 0, 0], [], []).unwrap();
        let run = MetalRun {
            elapsed: Duration::from_millis(5),
            spike_counts: vec![2, 0],
            voltage_mv: vec![-52.0; 2],
            conductance_mv: vec![0.0; 2],
        };
        let metadata = NativeRunMetadata {
            parameters: ModelParameters::default(),
            steps: 10,
            rate_hz: 150.0,
            seed: 7,
            stimulus_target_count: 1,
            missing_stimulus_ids: &[],
            chunk_steps: 4,
            device_name: "test",
            allocated_bytes: 100,
        };

        let manifest = write_native_run(&output, &connectome, &run, metadata).unwrap();

        assert_eq!(
            npy::read_u64(output.join("neuron_ids.npy")).unwrap(),
            [10, 20]
        );
        assert_eq!(manifest["total_spikes"], 2);
        assert_eq!(manifest["engine"], "rust-objc2-metal");
        assert!(output.join("manifest.json").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
