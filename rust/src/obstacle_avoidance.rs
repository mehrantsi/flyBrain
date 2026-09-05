use std::error::Error;
use std::fmt::{Display, Formatter};

const DEFAULT_BOUNDARY_BUFFER_MM: f64 = 0.0;
const DEFAULT_OBSTACLE_TRIGGER_MM: f64 = 55.0;
const DEFAULT_OBSTACLE_RELEASE_MM: f64 = 95.0;
const DEFAULT_ESCAPE_CLEAR_DWELL_SECONDS: f64 = 0.75;
const ESCAPE_REPLAN_SECONDS: f64 = 0.4;
const COLLISION_REFLEX_TRIGGER_MM: f64 = 8.0;
const EPSILON: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationObservation {
    pub position_mm: [f64; 3],
    pub forward_xy: [f64; 2],
    pub room_half_extents_mm: [f64; 3],
    pub forward_clearance_mm: f64,
    pub left_clearance_mm: f64,
    pub right_clearance_mm: f64,
    pub up_clearance_mm: f64,
    pub overhead: bool,
    pub dt_seconds: f64,
}

impl Default for NavigationObservation {
    fn default() -> Self {
        Self {
            position_mm: [0.0, 0.0, 28.0],
            forward_xy: [1.0, 0.0],
            room_half_extents_mm: [300.0, 220.0, 110.0],
            forward_clearance_mm: 1_000.0,
            left_clearance_mm: 1_000.0,
            right_clearance_mm: 1_000.0,
            up_clearance_mm: 1_000.0,
            overhead: false,
            dt_seconds: 0.02,
        }
    }
}

impl NavigationObservation {
    fn validate(self) -> Result<Self, NavigationError> {
        if self.position_mm.iter().any(|value| !value.is_finite())
            || self.forward_xy.iter().any(|value| !value.is_finite())
            || self
                .room_half_extents_mm
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || [
                self.forward_clearance_mm,
                self.left_clearance_mm,
                self.right_clearance_mm,
                self.up_clearance_mm,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0)
            || !self.dt_seconds.is_finite()
            || self.dt_seconds <= 0.0
        {
            return Err(NavigationError("navigation observation is invalid"));
        }
        if self.forward_xy[0].hypot(self.forward_xy[1]) <= EPSILON {
            return Err(NavigationError("navigation forward vector is zero"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EscapeSide {
    #[default]
    None,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationDecision {
    pub direction_xy: [f64; 2],
    pub steering: f64,
    pub boundary_steering: f64,
    pub obstacle_steering: f64,
    pub boundary_active: bool,
    pub obstacle_active: bool,
    pub collision_reflex_active: bool,
    pub escape_active: bool,
    pub escape_side: EscapeSide,
    pub altitude_escape: bool,
}

impl Default for NavigationDecision {
    fn default() -> Self {
        Self {
            direction_xy: [1.0, 0.0],
            steering: 0.0,
            boundary_steering: 0.0,
            obstacle_steering: 0.0,
            boundary_active: false,
            obstacle_active: false,
            collision_reflex_active: false,
            escape_active: false,
            escape_side: EscapeSide::None,
            altitude_escape: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationPolicyParameters {
    pub boundary_buffer_mm: f64,
    pub obstacle_trigger_mm: f64,
    pub obstacle_release_mm: f64,
    pub escape_clear_dwell_seconds: f64,
}

impl Default for NavigationPolicyParameters {
    fn default() -> Self {
        Self {
            boundary_buffer_mm: DEFAULT_BOUNDARY_BUFFER_MM,
            obstacle_trigger_mm: DEFAULT_OBSTACLE_TRIGGER_MM,
            obstacle_release_mm: DEFAULT_OBSTACLE_RELEASE_MM,
            escape_clear_dwell_seconds: DEFAULT_ESCAPE_CLEAR_DWELL_SECONDS,
        }
    }
}

impl NavigationPolicyParameters {
    fn validate(self) -> Result<Self, NavigationError> {
        if !self.boundary_buffer_mm.is_finite()
            || self.boundary_buffer_mm < 0.0
            || [
                self.obstacle_trigger_mm,
                self.obstacle_release_mm,
                self.escape_clear_dwell_seconds,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
            || self.obstacle_release_mm < self.obstacle_trigger_mm
        {
            return Err(NavigationError("navigation policy parameters are invalid"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationPolicy {
    parameters: NavigationPolicyParameters,
    escape_side: EscapeSide,
    escape_direction_xy: [f64; 2],
    escape_clear_elapsed_seconds: f64,
    escape_blocked_elapsed_seconds: f64,
}

impl Default for NavigationPolicy {
    fn default() -> Self {
        Self::with_parameters(NavigationPolicyParameters::default())
            .expect("default navigation policy parameters are valid")
    }
}

impl NavigationPolicy {
    pub fn with_parameters(
        parameters: NavigationPolicyParameters,
    ) -> Result<Self, NavigationError> {
        Ok(Self {
            parameters: parameters.validate()?,
            escape_side: EscapeSide::None,
            escape_direction_xy: [1.0, 0.0],
            escape_clear_elapsed_seconds: 0.0,
            escape_blocked_elapsed_seconds: 0.0,
        })
    }

    pub fn parameters(self) -> NavigationPolicyParameters {
        self.parameters
    }

    pub fn reset(&mut self) {
        self.escape_side = EscapeSide::None;
        self.escape_direction_xy = [1.0, 0.0];
        self.escape_clear_elapsed_seconds = 0.0;
        self.escape_blocked_elapsed_seconds = 0.0;
    }

    pub fn escape_side(&self) -> EscapeSide {
        self.escape_side
    }

    pub fn update(
        &mut self,
        observation: NavigationObservation,
    ) -> Result<NavigationDecision, NavigationError> {
        let observation = observation.validate()?;
        let forward = normalize_xy(observation.forward_xy);
        let left = [-forward[1], forward[0]];
        let right = [forward[1], -forward[0]];

        let (boundary_direction, boundary_steering, boundary_active) =
            self.boundary_response(observation, forward);

        let overhead_blocked = observation.overhead
            && observation.up_clearance_mm <= self.parameters.obstacle_trigger_mm;
        let forward_blocked =
            observation.forward_clearance_mm <= self.parameters.obstacle_trigger_mm;
        let obstacle_blocked = overhead_blocked || forward_blocked;

        if self.escape_side == EscapeSide::None && obstacle_blocked {
            self.escape_side = clearest_side(
                observation.left_clearance_mm,
                observation.right_clearance_mm,
            );
            let side = match self.escape_side {
                EscapeSide::Left => left,
                EscapeSide::Right => right,
                EscapeSide::None => unreachable!("escape side was just selected"),
            };
            self.escape_direction_xy = side;
            self.escape_clear_elapsed_seconds = 0.0;
            self.escape_blocked_elapsed_seconds = 0.0;
        }

        let escape_active = self.escape_side != EscapeSide::None;
        if escape_active && forward_blocked {
            self.escape_blocked_elapsed_seconds += observation.dt_seconds;
            if self.escape_blocked_elapsed_seconds >= ESCAPE_REPLAN_SECONDS {
                self.escape_side = clearest_side(
                    observation.left_clearance_mm,
                    observation.right_clearance_mm,
                );
                self.escape_direction_xy = match self.escape_side {
                    EscapeSide::Left => left,
                    EscapeSide::Right => right,
                    EscapeSide::None => unreachable!("escape side was just selected"),
                };
                self.escape_blocked_elapsed_seconds = 0.0;
            }
        } else {
            self.escape_blocked_elapsed_seconds = 0.0;
        }
        if escape_active {
            let clear = !overhead_blocked
                && !forward_blocked
                && observation.up_clearance_mm >= self.parameters.obstacle_release_mm
                && observation.forward_clearance_mm >= self.parameters.obstacle_release_mm;
            if clear {
                self.escape_clear_elapsed_seconds += observation.dt_seconds;
                if self.escape_clear_elapsed_seconds >= self.parameters.escape_clear_dwell_seconds {
                    self.escape_side = EscapeSide::None;
                    self.escape_clear_elapsed_seconds = 0.0;
                    self.escape_blocked_elapsed_seconds = 0.0;
                }
            } else {
                self.escape_clear_elapsed_seconds = 0.0;
            }
        }

        let escape_active = self.escape_side != EscapeSide::None;
        let (direction_xy, obstacle_steering) = if escape_active {
            (
                self.escape_direction_xy,
                signed_steering(forward, self.escape_direction_xy),
            )
        } else {
            (boundary_direction, 0.0)
        };

        let steering = clamp_unit(boundary_steering + obstacle_steering);
        let obstacle_active = obstacle_blocked || escape_active;
        let collision_reflex_active = (forward_blocked
            && observation.forward_clearance_mm <= COLLISION_REFLEX_TRIGGER_MM)
            || (overhead_blocked && observation.up_clearance_mm <= COLLISION_REFLEX_TRIGGER_MM);

        Ok(NavigationDecision {
            direction_xy: normalize_xy(direction_xy),
            steering,
            boundary_steering: finite_or_zero(boundary_steering),
            obstacle_steering: finite_or_zero(obstacle_steering),
            boundary_active,
            obstacle_active,
            collision_reflex_active,
            escape_active,
            escape_side: self.escape_side,
            altitude_escape: escape_active && overhead_blocked,
        })
    }

    fn boundary_response(
        &self,
        observation: NavigationObservation,
        forward: [f64; 2],
    ) -> ([f64; 2], f64, bool) {
        if self.parameters.boundary_buffer_mm <= EPSILON {
            return (forward, 0.0, false);
        }
        let distances = [
            observation.room_half_extents_mm[0] - observation.position_mm[0].abs(),
            observation.room_half_extents_mm[1] - observation.position_mm[1].abs(),
        ];
        let mut inward = [0.0, 0.0];
        for axis in 0..2 {
            let urgency = smoothstep(
                (self.parameters.boundary_buffer_mm - distances[axis])
                    / self.parameters.boundary_buffer_mm,
            );
            if urgency > 0.0 {
                let sign = if observation.position_mm[axis].is_sign_negative() {
                    1.0
                } else if observation.position_mm[axis] > 0.0 {
                    -1.0
                } else {
                    0.0
                };
                inward[axis] = sign * urgency;
            }
        }
        let inward_norm = inward[0].hypot(inward[1]);
        if inward_norm <= EPSILON {
            return (forward, 0.0, false);
        }
        let inward_direction = [inward[0] / inward_norm, inward[1] / inward_norm];
        let outward_direction = [-inward_direction[0], -inward_direction[1]];
        let heading_outward = dot_xy(forward, outward_direction).max(0.0);
        if heading_outward <= EPSILON {
            return (forward, 0.0, false);
        }
        let urgency = (inward_norm.min(1.0) * heading_outward).clamp(0.0, 1.0);
        let direction = normalize_xy_or(
            [
                (1.0 - urgency) * forward[0] + urgency * inward_direction[0],
                (1.0 - urgency) * forward[1] + urgency * inward_direction[1],
            ],
            inward_direction,
        );
        (direction, signed_steering(forward, direction), true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavigationError(&'static str);

impl Display for NavigationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for NavigationError {}

fn clearest_side(left_clearance_mm: f64, right_clearance_mm: f64) -> EscapeSide {
    if left_clearance_mm >= right_clearance_mm {
        EscapeSide::Left
    } else {
        EscapeSide::Right
    }
}

fn smoothstep(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn dot_xy(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

fn signed_steering(forward: [f64; 2], direction: [f64; 2]) -> f64 {
    let cross = forward[0] * direction[1] - forward[1] * direction[0];
    let dot = dot_xy(forward, direction);
    if cross.abs() <= EPSILON && dot < 0.0 {
        1.0
    } else {
        (cross.atan2(dot) / std::f64::consts::PI).clamp(-1.0, 1.0)
    }
}

fn normalize_xy(vector: [f64; 2]) -> [f64; 2] {
    normalize_xy_or(vector, [1.0, 0.0])
}

fn normalize_xy_or(vector: [f64; 2], fallback: [f64; 2]) -> [f64; 2] {
    let norm = vector[0].hypot(vector[1]);
    if norm.is_finite() && norm > EPSILON {
        [vector[0] / norm, vector[1] / norm]
    } else {
        normalize_xy_fallback(fallback)
    }
}

fn normalize_xy_fallback(vector: [f64; 2]) -> [f64; 2] {
    let norm = vector[0].hypot(vector[1]);
    if norm.is_finite() && norm > EPSILON {
        [vector[0] / norm, vector[1] / norm]
    } else {
        [1.0, 0.0]
    }
}

fn clamp_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else if value.is_sign_negative() {
        -1.0
    } else {
        1.0
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_observation() -> NavigationObservation {
        NavigationObservation::default()
    }

    #[test]
    fn table_center_has_no_boundary_response() {
        let mut policy = NavigationPolicy::default();
        let decision = policy
            .update(NavigationObservation {
                position_mm: [120.0, 70.0, 28.0],
                ..clear_observation()
            })
            .unwrap();
        assert!(!decision.boundary_active);
        assert_eq!(decision.boundary_steering, 0.0);
        assert_eq!(decision.direction_xy, [1.0, 0.0]);
    }

    #[test]
    fn default_policy_allows_approaching_room_walls() {
        let mut policy = NavigationPolicy::default();
        let decision = policy
            .update(NavigationObservation {
                position_mm: [-295.0, 0.0, 28.0],
                forward_xy: [-1.0, 0.0],
                ..clear_observation()
            })
            .unwrap();
        assert!(!decision.boundary_active);
        assert_eq!(decision.direction_xy, [-1.0, 0.0]);
    }

    #[test]
    fn near_wall_outward_heading_gets_inward_correction() {
        let mut policy = NavigationPolicy::with_parameters(NavigationPolicyParameters {
            boundary_buffer_mm: 80.0,
            ..NavigationPolicyParameters::default()
        })
        .unwrap();
        let decision = policy
            .update(NavigationObservation {
                position_mm: [-295.0, 0.0, 28.0],
                forward_xy: [-1.0, 0.0],
                ..clear_observation()
            })
            .unwrap();
        assert!(decision.boundary_active);
        assert!(decision.boundary_steering.abs() > 0.1);
        assert!(decision.direction_xy[0] > -1.0);
        assert!(decision.direction_xy[0] > 0.0 || decision.direction_xy[1].abs() > 0.0);
    }

    #[test]
    fn near_wall_inward_heading_is_left_alone() {
        let mut policy = NavigationPolicy::with_parameters(NavigationPolicyParameters {
            boundary_buffer_mm: 80.0,
            ..NavigationPolicyParameters::default()
        })
        .unwrap();
        let decision = policy
            .update(NavigationObservation {
                position_mm: [-295.0, 0.0, 28.0],
                forward_xy: [1.0, 0.0],
                ..clear_observation()
            })
            .unwrap();
        assert!(!decision.boundary_active);
        assert_eq!(decision.boundary_steering, 0.0);
        assert_eq!(decision.direction_xy, [1.0, 0.0]);
    }

    #[test]
    fn boundary_midpoint_never_produces_a_zero_direction() {
        let mut policy = NavigationPolicy::with_parameters(NavigationPolicyParameters {
            boundary_buffer_mm: 80.0,
            ..NavigationPolicyParameters::default()
        })
        .unwrap();
        let decision = policy
            .update(NavigationObservation {
                position_mm: [-130.0, 0.0, 28.0],
                forward_xy: [-1.0, 0.0],
                ..clear_observation()
            })
            .unwrap();
        assert!(decision.direction_xy[0].hypot(decision.direction_xy[1]) > 0.99);
        assert!(decision.direction_xy.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn overhang_selects_clearest_side_and_holds_it_until_clear() {
        let mut policy = NavigationPolicy::default();
        let blocked = NavigationObservation {
            position_mm: [120.0, 70.0, 28.0],
            forward_clearance_mm: 20.0,
            left_clearance_mm: 110.0,
            right_clearance_mm: 35.0,
            up_clearance_mm: 16.0,
            overhead: true,
            ..clear_observation()
        };
        let first = policy.update(blocked).unwrap();
        assert!(first.obstacle_active);
        assert!(first.escape_active);
        assert!(first.altitude_escape);
        assert_eq!(first.escape_side, EscapeSide::Left);
        assert!(first.direction_xy[1] > 0.0);

        let mut changing_side = blocked;
        changing_side.left_clearance_mm = 20.0;
        changing_side.right_clearance_mm = 130.0;
        let held = policy.update(changing_side).unwrap();
        assert!(held.escape_active);
        assert_eq!(held.escape_side, EscapeSide::Left);
        assert!(held.direction_xy[1] > 0.0);

        changing_side.forward_xy = [0.0, 1.0];
        let held_world_direction = policy.update(changing_side).unwrap();
        assert_eq!(held_world_direction.direction_xy, first.direction_xy);

        let mut clear = clear_observation();
        clear.dt_seconds = 0.25;
        clear.forward_clearance_mm = 120.0;
        clear.left_clearance_mm = 120.0;
        clear.right_clearance_mm = 120.0;
        clear.up_clearance_mm = 120.0;
        clear.overhead = false;
        assert!(policy.update(clear).unwrap().escape_active);
        assert!(policy.update(clear).unwrap().escape_active);
        assert!(!policy.update(clear).unwrap().escape_active);
        assert_eq!(policy.escape_side(), EscapeSide::None);
    }

    #[test]
    fn clear_observation_has_finite_nonzero_output() {
        let mut policy = NavigationPolicy::default();
        let decision = policy.update(clear_observation()).unwrap();
        assert!(decision.direction_xy.iter().all(|value| value.is_finite()));
        assert!(decision.direction_xy[0].hypot(decision.direction_xy[1]) > 0.99);
        assert!(decision.steering.is_finite());
        assert!(decision.boundary_steering.is_finite());
        assert!(decision.obstacle_steering.is_finite());
    }

    #[test]
    fn collision_reflex_is_reserved_for_immediate_clearance() {
        let mut policy = NavigationPolicy::default();
        let near = policy
            .update(NavigationObservation {
                forward_clearance_mm: 20.0,
                ..clear_observation()
            })
            .unwrap();
        assert!(near.obstacle_active);
        assert!(!near.collision_reflex_active);

        let immediate = policy
            .update(NavigationObservation {
                forward_clearance_mm: 5.0,
                ..clear_observation()
            })
            .unwrap();
        assert!(immediate.collision_reflex_active);
    }

    #[test]
    fn blocked_escape_path_replans_after_the_dwell() {
        let mut policy = NavigationPolicy::default();
        let mut blocked = NavigationObservation {
            forward_clearance_mm: 20.0,
            left_clearance_mm: 120.0,
            right_clearance_mm: 30.0,
            up_clearance_mm: 20.0,
            overhead: true,
            dt_seconds: 0.2,
            ..clear_observation()
        };
        let first = policy.update(blocked).unwrap();
        assert_eq!(first.escape_side, EscapeSide::Left);

        blocked.left_clearance_mm = 25.0;
        blocked.right_clearance_mm = 130.0;
        let replanned = policy.update(blocked).unwrap();
        assert_eq!(replanned.escape_side, EscapeSide::Right);
        assert!(replanned.direction_xy[1] < 0.0);
    }
}
