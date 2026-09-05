from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def motor_spikes(report):
    return sum(group["total_spikes"] for group in report["readouts"].values())


def summarize(reports):
    rows = []
    checks = []
    for seed in sorted(reports):
        runs = reports[seed]
        intact = runs["intact"]
        no_input = runs["no-input"]
        disconnected = runs["input-disconnected"]
        relay_off = runs["relay-disconnected"]
        for control, report in runs.items():
            rows.append({
                "seed": seed, "control": control,
                "input_spikes": report["inputs"]["total_spikes"],
                "relay_spikes": report["relays"]["total_spikes"],
                "motor_spikes": motor_spikes(report),
                "population_spikes": report["population"]["total_spikes"],
            })
        checks.append({
            "seed": seed,
            "no_input_is_quiet": no_input["population"]["total_spikes"] == 0,
            "input_disconnect_preserves_stimulus": (
                disconnected["stimulus"]["event_count"] == intact["stimulus"]["event_count"]
                and disconnected["inputs"]["total_spikes"] > 0
            ),
            "input_disconnect_blocks_all_downstream_spikes": (
                disconnected["population"]["total_spikes"]
                == disconnected["inputs"]["total_spikes"]
            ),
            "intact_reaches_relay_and_motors": (
                intact["relays"]["total_spikes"] > 0 and motor_spikes(intact) > 0
            ),
            "both_relays_active": intact["relays"]["active_neurons"] == 2,
            "all_six_motor_pools_active": (
                len(intact["readouts"]) == 6
                and all(group["total_spikes"] > 0 for group in intact["readouts"].values())
            ),
            "all_twelve_motor_neurons_active": (
                sum(group["active_neurons"] for group in intact["readouts"].values()) == 12
            ),
            "relay_disconnect_reduces_motor_spikes": (
                motor_spikes(relay_off) < motor_spikes(intact)
            ),
            "relay_disconnect_motor_reduction_fraction": (
                1 - motor_spikes(relay_off) / motor_spikes(intact)
                if motor_spikes(intact) else None
            ),
        })
    return {
        "rows": rows, "checks": checks,
        "software_controls_pass": all(
            row["no_input_is_quiet"]
            and row["input_disconnect_preserves_stimulus"]
            and row["input_disconnect_blocks_all_downstream_spikes"]
            for row in checks
        ),
        "pathway_response_reproduced_all_seeds": all(
            row["intact_reaches_relay_and_motors"]
            and row["relay_disconnect_reduces_motor_spikes"]
            for row in checks
        ),
        "interpretation": (
            "These are computational intervention checks, not validated landing behavior. "
            "Relay output disconnection removes outgoing synapses, not alternate network paths. "
            "Stimulus intensity, cell dynamics and motor-to-muscle mapping are not calibrated."
        ),
        "live_default_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Run matched MaleCNS pathway interventions")
    parser.add_argument("--binary", type=Path, default=Path("target/release/flybrain-rs"))
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--pathway", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for seed in (1, 2, 3):
        reports[seed] = {}
        for control in (
            "intact", "no-input", "input-disconnected", "relay-disconnected", "relay-driven"
        ):
            output = args.output / f"seed-{seed}-{control}.json"
            command = [
                str(args.binary.resolve()), "pathway", "--pack", str(args.pack),
                "--pathway", str(args.pathway), "--control", control,
                "--steps", "2000", "--rate-hz", "150", "--seed", str(seed),
                "--output", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            reports[seed][control] = json.loads(output.read_text())
            print(f"seed {seed}: {control} complete", flush=True)
    parity_path = args.output / "cpu-parity-100ms.json"
    subprocess.run([
        str(args.binary.resolve()), "pathway", "--pack", str(args.pack),
        "--pathway", str(args.pathway), "--control", "intact", "--steps", "1000",
        "--rate-hz", "150", "--seed", "1", "--verify-cpu", "--output", str(parity_path),
    ], check=True, capture_output=True, text=True)
    summary = summarize(reports)
    short_parity_path = args.output / "cpu-parity-20ms.json"
    subprocess.run([
        str(args.binary.resolve()), "pathway", "--pack", str(args.pack),
        "--pathway", str(args.pathway), "--control", "intact", "--steps", "200",
        "--rate-hz", "150", "--seed", "1", "--verify-cpu", "--output", str(short_parity_path),
    ], check=True, capture_output=True, text=True)
    summary["cpu_short_window_verification"] = json.loads(
        short_parity_path.read_text()
    )["cpu_verification"]
    summary["cpu_verification"] = json.loads(parity_path.read_text())["cpu_verification"]
    cpu = summary["cpu_verification"]
    summary["cpu_check_tolerance_mv"] = 0.001
    summary["cpu_numerical_check_pass"] = (
        cpu["spike_counts_exact"]
        and cpu["maximum_voltage_error_mv"] <= summary["cpu_check_tolerance_mv"]
        and cpu["maximum_conductance_error_mv"] <= summary["cpu_check_tolerance_mv"]
    )
    summary["acceptance_pass"] = (
        summary["software_controls_pass"]
        and summary["pathway_response_reproduced_all_seeds"]
        and summary["cpu_numerical_check_pass"]
    )
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    if not summary["acceptance_pass"]:
        raise RuntimeError(f"CNS experiment gate failed; see {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
