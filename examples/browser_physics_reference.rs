use anyhow::{Result, ensure};
use flybrain_engine::world_sim::SimulationStepper;
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let output = std::env::args().nth(1).expect("provide a new output path");
    ensure!(
        !Path::new(&output).exists(),
        "refusing to replace an existing reference"
    );
    let mut sim = SimulationStepper::new("assets/neuromechfly", None::<&Path>, 500.0, 0.5)?;
    sim.place_food_ahead(40.0)?;
    let mut frames = Vec::new();
    for i in 0..=10 {
        if i != 0 {
            sim.step_window()?;
        }
        frames.push(json!({"time_seconds":sim.snapshot().time_seconds,
            "qpos":sim.world().qpos(), "qvel":sim.world().qvel()}));
    }
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({"frames":frames}))?,
    )?;
    println!("Saved native MuJoCo controller trace: {output}");
    Ok(())
}
