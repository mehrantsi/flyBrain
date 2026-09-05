pub mod aerodynamics;
pub mod behavior;
pub mod brain_signal;
pub mod cns_olfaction;
pub mod cns_pathway;
pub mod embodiment;
pub mod fixture;
pub mod flight;
pub mod flight_behavior;
pub mod flight_targets;
pub mod flybody_policy;
pub mod foraging;
pub mod gait;
pub mod grooming;
pub mod habitat;
pub mod neural_io;
pub mod npy;
pub mod obstacle_avoidance;
pub mod odor_guidance;
pub mod olfaction;
pub mod pack;
pub mod parameters;
pub mod protocol;
pub mod reference;
pub mod stimulus;
pub mod system_id;

#[cfg(target_os = "macos")]
pub mod output;

#[cfg(any(target_os = "macos", target_os = "emscripten"))]
pub mod brain_bridge;

#[cfg(target_os = "macos")]
pub mod metal_engine;

#[cfg(target_os = "emscripten")]
pub mod browser_engine;

#[cfg(target_os = "emscripten")]
pub mod browser;

#[cfg(any(target_os = "macos", target_os = "emscripten"))]
pub mod world;

#[cfg(target_os = "macos")]
pub mod render;

#[cfg(target_os = "macos")]
pub mod live_viewer;

#[cfg(any(target_os = "macos", target_os = "emscripten"))]
pub mod retina;

#[cfg(any(target_os = "macos", target_os = "emscripten"))]
pub mod world_sim;

#[cfg(target_os = "macos")]
pub mod flight_system_id_world;
