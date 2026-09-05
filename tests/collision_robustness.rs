#![cfg(target_os = "macos")]

use std::os::raw::c_char;
use std::sync::atomic::{AtomicUsize, Ordering};

use mujoco_rs::mujoco_c;
use mujoco_rs::prelude::{MjModel, MjtObj};
use serde::Deserialize;

const MODEL_PATH: &str = "assets/neuromechfly/fly.xml";
const CCD_TOLERANCE: f64 = 1e-6;
const CONFIGURED_CCD_ITERATIONS: i32 = 100;
const HIGH_BUDGET_CCD_ITERATIONS: i32 = 200;
const CONVERGENCE_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Deserialize)]
struct ReplayFixture {
    schema: String,
    source: ReplaySource,
    states: Vec<ReplayState>,
}

#[derive(Debug, Deserialize)]
struct ReplaySource {
    model: String,
    states: Vec<String>,
    legacy_replay: LegacyReplay,
}

#[derive(Debug, Deserialize)]
struct LegacyReplay {
    ccd_tolerance: f64,
    ccd_iterations: i32,
    expected_warnings: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ReplayState {
    id: String,
    time: f64,
    geom1: String,
    geom2: String,
    margin: f64,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct ContactSnapshot {
    dist: f64,
    pos: [f64; 3],
    normal: [f64; 3],
}

#[derive(Debug)]
struct ReplayResult {
    warning_counts: Vec<usize>,
    contacts: Vec<ContactSnapshot>,
}

static WARNING_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn capture_warning(_: *const c_char) {
    WARNING_COUNT.fetch_add(1, Ordering::SeqCst);
}

struct WarningCallbackGuard {
    previous: Option<unsafe extern "C" fn(*const c_char)>,
}

impl WarningCallbackGuard {
    fn install() -> Self {
        let previous = unsafe { mujoco_c::mju_user_warning };
        unsafe {
            mujoco_c::mju_user_warning = Some(capture_warning);
        }
        Self { previous }
    }

    fn reset_count() {
        WARNING_COUNT.store(0, Ordering::SeqCst);
    }

    fn count() -> usize {
        WARNING_COUNT.load(Ordering::SeqCst)
    }
}

impl Drop for WarningCallbackGuard {
    fn drop(&mut self) {
        unsafe {
            mujoco_c::mju_user_warning = self.previous;
        }
    }
}

fn fixture() -> ReplayFixture {
    serde_json::from_str(include_str!("fixtures/cylinder_mesh_departure.json"))
        .expect("EPA replay fixture must be valid JSON")
}

fn model_with_ccd_iterations(iterations: i32) -> MjModel {
    let mut model = MjModel::from_xml(MODEL_PATH).expect("actual model should load");
    model.opt_mut().ccd_tolerance = CCD_TOLERANCE;
    model.opt_mut().ccd_iterations = iterations;
    model
}

fn replay(model: &MjModel, fixture: &ReplayFixture) -> ReplayResult {
    let geom1_ids: Vec<_> = fixture
        .states
        .iter()
        .map(|state| {
            model
                .name_to_id(MjtObj::mjOBJ_GEOM, &state.geom1)
                .expect("fixture geom1 must be present")
        })
        .collect();
    let geom2_ids: Vec<_> = fixture
        .states
        .iter()
        .map(|state| {
            model
                .name_to_id(MjtObj::mjOBJ_GEOM, &state.geom2)
                .expect("fixture geom2 must be present")
        })
        .collect();
    let mut data = model.make_data();
    let mut warning_counts = Vec::with_capacity(fixture.states.len());
    let mut contacts = Vec::with_capacity(fixture.states.len());

    for (index, state) in fixture.states.iter().enumerate() {
        assert_eq!(state.qpos.len(), model.nq() as usize, "{} qpos", state.id);
        assert_eq!(state.qvel.len(), model.nv() as usize, "{} qvel", state.id);
        data.qpos_mut().copy_from_slice(&state.qpos);
        data.qvel_mut().copy_from_slice(&state.qvel);
        data.set_time(state.time);
        WarningCallbackGuard::reset_count();
        data.forward();
        warning_counts.push(WarningCallbackGuard::count());

        let expected_geom1 = geom1_ids[index] as i32;
        let expected_geom2 = geom2_ids[index] as i32;
        let matches: Vec<_> = data
            .contact()
            .iter()
            .filter(|contact| {
                (contact.geom1 == expected_geom1 && contact.geom2 == expected_geom2)
                    || (contact.geom1 == expected_geom2 && contact.geom2 == expected_geom1)
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "{} expected one captured contact",
            state.id
        );
        let contact = matches[0];
        assert!(contact.dist.is_finite(), "{} contact distance", state.id);
        assert!(
            contact.pos.iter().all(|value| value.is_finite()),
            "{} contact position",
            state.id
        );
        assert!(
            contact.frame.iter().all(|value| value.is_finite()),
            "{} contact frame",
            state.id
        );
        let normal = [contact.frame[0], contact.frame[1], contact.frame[2]];
        assert!(
            normal.iter().all(|value| value.is_finite()),
            "{} contact normal",
            state.id
        );
        assert!(
            normal.iter().map(|value| value * value).sum::<f64>() > 0.5,
            "{} contact normal",
            state.id
        );
        assert!((contact.includemargin - state.margin).abs() <= f64::EPSILON);
        contacts.push(ContactSnapshot {
            dist: contact.dist,
            pos: contact.pos,
            normal,
        });
    }

    ReplayResult {
        warning_counts,
        contacts,
    }
}

fn assert_converged(left: &[ContactSnapshot], right: &[ContactSnapshot]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left.dist - right.dist).abs() <= CONVERGENCE_TOLERANCE,
            "state {index} dist"
        );
        for axis in 0..3 {
            assert!(
                (left.pos[axis] - right.pos[axis]).abs() <= CONVERGENCE_TOLERANCE,
                "state {index} pos[{axis}]"
            );
            assert!(
                (left.normal[axis] - right.normal[axis]).abs() <= CONVERGENCE_TOLERANCE,
                "state {index} normal[{axis}]"
            );
        }
    }
}

#[test]
fn configured_ccd_replays_cylinder_mesh_departure_without_epa_warnings() {
    let fixture = fixture();
    assert_eq!(fixture.schema, "flybrain-mujoco-epa-replay-v1");
    assert_eq!(fixture.source.model, MODEL_PATH);
    assert_eq!(fixture.source.states.len(), fixture.states.len());
    assert_eq!(fixture.source.legacy_replay.ccd_tolerance, CCD_TOLERANCE);

    let configured_model = model_with_ccd_iterations(CONFIGURED_CCD_ITERATIONS);
    assert_eq!(
        configured_model.opt().ccd_iterations,
        CONFIGURED_CCD_ITERATIONS
    );
    let legacy_model = model_with_ccd_iterations(fixture.source.legacy_replay.ccd_iterations);
    let high_budget_model = model_with_ccd_iterations(HIGH_BUDGET_CCD_ITERATIONS);

    let _warning_guard = WarningCallbackGuard::install();
    let legacy = replay(&legacy_model, &fixture);
    assert_eq!(
        legacy.warning_counts,
        fixture.source.legacy_replay.expected_warnings
    );
    let configured = replay(&configured_model, &fixture);
    assert_eq!(configured.warning_counts, vec![0; fixture.states.len()]);
    let high_budget = replay(&high_budget_model, &fixture);
    assert_eq!(high_budget.warning_counts, vec![0; fixture.states.len()]);
    assert_converged(&configured.contacts, &high_budget.contacts);
}
