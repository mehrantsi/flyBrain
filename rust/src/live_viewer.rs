use std::collections::{BTreeSet, VecDeque};
use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

use anyhow::{Context, Result, bail};
use mujoco_rs::mujoco_c::{mjr_blitBuffer, mjr_drawPixels, mjr_label};
use mujoco_rs::prelude::*;

use crate::render::RoomPresentation;
use crate::retina::{FlyGymRetina, RETINA_HEIGHT, RETINA_WIDTH, RetinaSummary};

const GLFW_FALSE: c_int = 0;
const GLFW_TRUE: c_int = 1;
const GLFW_PRESS: c_int = 1;
const GLFW_VISIBLE: c_int = 0x0002_0004;
const GLFW_SAMPLES: c_int = 0x0002_100D;

const KEY_SPACE: c_int = 32;
const KEY_1: c_int = 49;
const KEY_2: c_int = 50;
const KEY_3: c_int = 51;
const KEY_4: c_int = 52;
const KEY_5: c_int = 53;
const KEY_6: c_int = 54;
const KEY_A: c_int = 65;
const KEY_B: c_int = 66;
const KEY_D: c_int = 68;
const KEY_F: c_int = 70;
const KEY_G: c_int = 71;
const KEY_H: c_int = 72;
const KEY_R: c_int = 82;
const KEY_S: c_int = 83;
const KEY_T: c_int = 84;
const KEY_V: c_int = 86;
const KEY_W: c_int = 87;
const KEY_ESCAPE: c_int = 256;

const MOUSE_LEFT: c_int = 0;
const MOUSE_RIGHT: c_int = 1;
const MOUSE_MIDDLE: c_int = 2;

pub const CHASE_CAMERA_NAME: &str = "chase";
const CHASE_BODY_NAME: &str = "fly/c_thorax";
const CHASE_DISTANCE: f64 = 36.0;
const CHASE_AZIMUTH_DEG: f64 = 135.0;
const CHASE_ELEVATION_DEG: f64 = -22.0;
const SIDE_CAMERA_DISTANCE: f64 = 28.0;
const SIDE_CAMERA_ELEVATION_DEG: f64 = -12.0;
const TRACKING_CAMERA_MIN_DISTANCE: f64 = 12.0;
const TRACKING_CAMERA_RESTORE_DISTANCE: f64 = 20.0;
const TRACKING_CAMERA_WALL_BOUNDS: [[f64; 2]; 3] =
    [[-297.0, 297.0], [-217.0, 217.0], [f64::NEG_INFINITY, 216.0]];
const TRACKING_CAMERA_AZIMUTH_OFFSETS: [f64; 4] = [0.0, 90.0, -90.0, 180.0];

#[link(name = "glfw.3")]
unsafe extern "C" {
    fn glfwInit() -> c_int;
    fn glfwTerminate();
    fn glfwWindowHint(hint: c_int, value: c_int);
    fn glfwCreateWindow(
        width: c_int,
        height: c_int,
        title: *const c_char,
        monitor: *mut c_void,
        share: *mut c_void,
    ) -> *mut c_void;
    fn glfwDestroyWindow(window: *mut c_void);
    fn glfwMakeContextCurrent(window: *mut c_void);
    fn glfwSwapInterval(interval: c_int);
    fn glfwSwapBuffers(window: *mut c_void);
    fn glfwPollEvents();
    fn glfwWindowShouldClose(window: *mut c_void) -> c_int;
    fn glfwSetWindowShouldClose(window: *mut c_void, value: c_int);
    fn glfwGetFramebufferSize(window: *mut c_void, width: *mut c_int, height: *mut c_int);
    fn glfwGetWindowSize(window: *mut c_void, width: *mut c_int, height: *mut c_int);
    fn glfwGetKey(window: *mut c_void, key: c_int) -> c_int;
    fn glfwGetMouseButton(window: *mut c_void, button: c_int) -> c_int;
    fn glfwGetCursorPos(window: *mut c_void, x: *mut f64, y: *mut f64);
    fn glfwSetWindowTitle(window: *mut c_void, title: *const c_char);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LiveInput {
    pub quit: bool,
    pub toggle_pause: bool,
    pub reset: bool,
    pub toggle_food: bool,
    pub toggle_flight: bool,
    pub request_grooming: bool,
    pub place_food_at_mouth: bool,
    pub toggle_eye_view: bool,
    pub toggle_brain_graph: bool,
    pub food_motion: [f64; 2],
}

pub struct LiveRenderOptions<'a> {
    pub food_center: [f64; 3],
    pub food_enabled: bool,
    pub status: &'a str,
    pub show_eye_view: bool,
    pub show_brain_graph: bool,
    pub capture_vision: bool,
    pub flight_allowed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TrackingCameraProfile {
    nominal_distance: f64,
    nominal_azimuth_deg: f64,
    elevation_deg: f64,
    azimuth_offset_deg: f64,
}

impl TrackingCameraProfile {
    fn new(distance: f64, azimuth_deg: f64, elevation_deg: f64) -> Self {
        Self {
            nominal_distance: distance,
            nominal_azimuth_deg: azimuth_deg,
            elevation_deg,
            azimuth_offset_deg: 0.0,
        }
    }
}

pub struct LiveViewer {
    window: NonNull<c_void>,
    context: Option<MjrContext>,
    scene: Option<MjvScene>,
    eye_scene: Option<MjvScene>,
    presentation: RoomPresentation,
    camera: MjvCamera,
    tracking_camera: Option<TrackingCameraProfile>,
    hide_fly_in_main_view: bool,
    eye_camera_ids: [usize; 2],
    option: MjvOption,
    eye_option: MjvOption,
    perturb: MjvPerturb,
    retinas: [FlyGymRetina; 2],
    retina_summaries: [RetinaSummary; 2],
    eye_raw_bottom_up: Box<[u8]>,
    eye_raw_top_down: Box<[u8]>,
    eye_display_bottom_up: [Box<[u8]>; 2],
    brain_figure: Box<MjvFigure>,
    brain_samples: VecDeque<(f32, f32)>,
    last_brain_sample_time: Option<f64>,
    brain_figure_dirty: bool,
    brain_dominant_frequency_hz: f64,
    food_geom_id: usize,
    pressed_keys: BTreeSet<c_int>,
    cursor: [f64; 2],
    cursor_initialized: bool,
    mouse_left_was_down: bool,
}

impl LiveViewer {
    pub fn new(
        model: &MjModel,
        assets_dir: impl AsRef<Path>,
        width: u32,
        height: u32,
        camera_name: &str,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("viewer dimensions must be positive")
        }
        let width = c_int::try_from(width).context("viewer width exceeds GLFW limits")?;
        let height = c_int::try_from(height).context("viewer height exceeds GLFW limits")?;
        let (camera, tracking_camera) = if camera_name == CHASE_CAMERA_NAME {
            (
                Self::new_chase_camera(model)?,
                Some(TrackingCameraProfile::new(
                    CHASE_DISTANCE,
                    CHASE_AZIMUTH_DEG,
                    CHASE_ELEVATION_DEG,
                )),
            )
        } else {
            let camera_id = model
                .name_to_id(MjtObj::mjOBJ_CAMERA, camera_name)
                .with_context(|| format!("model has no camera named {camera_name}"))?;
            (MjvCamera::new_fixed(camera_id), None)
        };
        let food_geom_id = model
            .name_to_id(MjtObj::mjOBJ_GEOM, "food_patch")
            .context("model has no food_patch geometry")?;
        let eye_camera_ids = ["fly/l_eye_cam_camera", "fly/r_eye_cam_camera"]
            .map(|name| {
                model
                    .name_to_id(MjtObj::mjOBJ_CAMERA, name)
                    .with_context(|| format!("model has no camera named {name}"))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("fly-eye camera list has the wrong shape"))?;
        let assets_dir = assets_dir.as_ref();
        let retinas = [
            FlyGymRetina::load(assets_dir)?,
            FlyGymRetina::load(assets_dir)?,
        ];
        let title = CString::new("FlyBrain live")?;
        let window = unsafe {
            if glfwInit() == GLFW_FALSE {
                bail!("GLFW initialization failed")
            }
            glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
            glfwWindowHint(GLFW_SAMPLES, 4);
            let window = NonNull::new(glfwCreateWindow(
                width,
                height,
                title.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ));
            let Some(window) = window else {
                glfwTerminate();
                bail!("GLFW window creation failed")
            };
            glfwMakeContextCurrent(window.as_ptr());
            glfwSwapInterval(0);
            window
        };
        let mut context = unsafe { MjrContext::new(model) };
        context.resize_offscreen(RETINA_WIDTH as u32, RETINA_HEIGHT as u32);
        context.window();
        let scene = MjvScene::new(model, model.ngeom() as usize + 64);
        let eye_scene = MjvScene::new(model, model.ngeom() as usize + 64);
        let mut eye_option = MjvOption::default();
        eye_option.geomgroup[1] = 0;
        eye_option.geomgroup[2] = 0;
        let eye_rgb_bytes = RETINA_WIDTH * RETINA_HEIGHT * 3;
        let mut brain_figure = MjvFigure::new_boxed();
        brain_figure.set_title("EEG-like field potential | NETWORK PROXY");
        brain_figure.set_xlabel("simulation time (s)");
        brain_figure.set_linename(0, "global mean membrane field (mV)");
        brain_figure.flg_legend = 1;
        brain_figure.flg_extend = 0;
        brain_figure.flg_symmetric = 1;
        brain_figure.gridsize = [5, 5];
        brain_figure.linergb[0] = [0.15, 0.9, 1.0];
        brain_figure.figurergba = [0.02, 0.03, 0.06, 0.92];
        brain_figure.panergba = [0.04, 0.06, 0.10, 0.92];
        brain_figure.legendrgba = [0.02, 0.03, 0.06, 0.86];
        brain_figure.textrgb = [0.9, 0.95, 1.0];
        let mut viewer = Self {
            window,
            context: Some(context),
            scene: Some(scene),
            eye_scene: Some(eye_scene),
            presentation: RoomPresentation::new(model),
            camera,
            tracking_camera,
            hide_fly_in_main_view: camera_name == "fly/l_eye_cam_camera"
                || camera_name == "fly/r_eye_cam_camera",
            eye_camera_ids,
            option: MjvOption::default(),
            eye_option,
            perturb: MjvPerturb::default(),
            retinas,
            retina_summaries: [RetinaSummary::default(); 2],
            eye_raw_bottom_up: vec![0; eye_rgb_bytes].into_boxed_slice(),
            eye_raw_top_down: vec![0; eye_rgb_bytes].into_boxed_slice(),
            eye_display_bottom_up: std::array::from_fn(|_| {
                vec![0; eye_rgb_bytes].into_boxed_slice()
            }),
            brain_figure,
            brain_samples: VecDeque::with_capacity(1_024),
            last_brain_sample_time: None,
            brain_figure_dirty: false,
            brain_dominant_frequency_hz: 0.0,
            food_geom_id,
            pressed_keys: BTreeSet::new(),
            cursor: [0.0; 2],
            cursor_initialized: false,
            mouse_left_was_down: false,
        };
        viewer.read_cursor();
        Ok(viewer)
    }

    pub fn is_open(&self) -> bool {
        unsafe { glfwWindowShouldClose(self.window.as_ptr()) == GLFW_FALSE }
    }

    pub fn poll_input(&mut self, model: &MjModel) -> LiveInput {
        unsafe { glfwPollEvents() };
        let quit = self.key_pressed(KEY_ESCAPE);
        if quit {
            unsafe { glfwSetWindowShouldClose(self.window.as_ptr(), GLFW_TRUE) };
        }
        if self.key_pressed(KEY_1) {
            self.use_chase_camera(model);
        }
        if self.key_pressed(KEY_2) {
            self.use_tracking_camera(model, SIDE_CAMERA_DISTANCE, 90.0, SIDE_CAMERA_ELEVATION_DEG);
        }
        if self.key_pressed(KEY_3) {
            self.use_tracking_camera(
                model,
                SIDE_CAMERA_DISTANCE,
                270.0,
                SIDE_CAMERA_ELEVATION_DEG,
            );
        }
        if self.key_pressed(KEY_4) {
            self.camera = MjvCamera::new_free(model);
            self.tracking_camera = None;
            self.hide_fly_in_main_view = false;
        }
        if self.key_pressed(KEY_5) {
            self.use_fixed_camera(model, "room_camera", false);
        }
        if self.key_pressed(KEY_6) {
            self.use_fixed_camera(model, "fly/trackingcam", false);
        }
        let (eye_button_clicked, brain_button_clicked, flight_button_clicked) =
            self.hud_button_clicks();
        self.update_camera_from_mouse(model);
        let move_scale = 0.025;
        LiveInput {
            quit,
            toggle_pause: self.key_pressed(KEY_SPACE),
            reset: self.key_pressed(KEY_R),
            toggle_food: self.key_pressed(KEY_F),
            toggle_flight: self.key_pressed(KEY_G) || flight_button_clicked,
            request_grooming: self.key_pressed(KEY_H),
            place_food_at_mouth: self.key_pressed(KEY_T),
            toggle_eye_view: self.key_pressed(KEY_V) || eye_button_clicked,
            toggle_brain_graph: self.key_pressed(KEY_B) || brain_button_clicked,
            food_motion: [
                move_scale * f64::from(self.key_down(KEY_W) as u8)
                    - move_scale * f64::from(self.key_down(KEY_S) as u8),
                move_scale * f64::from(self.key_down(KEY_A) as u8)
                    - move_scale * f64::from(self.key_down(KEY_D) as u8),
            ],
        }
    }

    pub fn render<M>(&mut self, data: &mut MjData<M>, options: LiveRenderOptions<'_>) -> Result<()>
    where
        M: std::ops::Deref<Target = MjModel>,
    {
        unsafe { glfwMakeContextCurrent(self.window.as_ptr()) };
        if options.show_brain_graph {
            self.refresh_brain_figure();
        }
        let scene = self.scene.as_mut().context("viewer scene is unavailable")?;
        if let Some(profile) = self.tracking_camera.as_mut() {
            update_collision_aware_tracking_scene(
                scene,
                data,
                &self.option,
                &self.perturb,
                &mut self.camera,
                profile,
            );
        } else {
            scene.update(data, &self.option, &self.perturb, &mut self.camera);
        }
        if self.hide_fly_in_main_view {
            hide_fly_visuals(scene, data.model());
        } else {
            self.presentation.apply(scene);
        }
        move_food_geom(
            scene,
            self.food_geom_id,
            options.food_center,
            options.food_enabled,
        );
        let mut width = 0;
        let mut height = 0;
        unsafe { glfwGetFramebufferSize(self.window.as_ptr(), &mut width, &mut height) };
        if width > 0 && height > 0 {
            let viewport = MjrRectangle::new(0, 0, width, height);
            let context = self
                .context
                .as_mut()
                .context("viewer render context is unavailable")?;
            context.window();
            scene.render(&viewport, context);
            context.overlay(
                MjtFont::mjFONT_NORMAL,
                MjtGridPos::mjGRID_TOPLEFT,
                viewport,
                options.status,
                None,
            );
            context.overlay(
                MjtFont::mjFONT_NORMAL,
                MjtGridPos::mjGRID_BOTTOMLEFT,
                viewport,
                &format!(
                    "SPACE pause  R reset  H groom  T drop sugar below fly  F food on/off  W/A/S/D move food\nV binocular ommatidial retina {}  B EEG proxy {}  G autonomous flight {}  1 chase  2/3 side views  4 free  5 room  6 close track  ESC quit",
                    on_off(options.show_eye_view),
                    on_off(options.show_brain_graph),
                    on_off(options.flight_allowed),
                ),
                None,
            );
            let [eye_button, brain_button, flight_button] = hud_button_rects(width, height);
            draw_hud_button(
                context,
                eye_button,
                &format!("V  RETINA: {}", on_off(options.show_eye_view)),
                options.show_eye_view,
            );
            draw_hud_button(
                context,
                brain_button,
                &format!("B  EEG PROXY: {}", on_off(options.show_brain_graph)),
                options.show_brain_graph,
            );
            draw_hud_button(
                context,
                flight_button,
                &format!("G  AUTO FLIGHT: {}", on_off(options.flight_allowed)),
                options.flight_allowed,
            );
            if options.show_eye_view || options.capture_vision {
                let eye_scene = self
                    .eye_scene
                    .as_mut()
                    .context("fly-eye scene is unavailable")?;
                let retina_viewport =
                    MjrRectangle::new(0, 0, RETINA_WIDTH as c_int, RETINA_HEIGHT as c_int);
                for eye_index in 0..2 {
                    let mut eye_camera = MjvCamera::new_fixed(self.eye_camera_ids[eye_index]);
                    eye_scene.update(data, &self.eye_option, &self.perturb, &mut eye_camera);
                    hide_fly_visuals(eye_scene, data.model());
                    move_food_geom(
                        eye_scene,
                        self.food_geom_id,
                        options.food_center,
                        options.food_enabled,
                    );
                    context.offscreen();
                    eye_scene.render(&retina_viewport, context);
                    context.read_pixels(
                        Some(&mut self.eye_raw_bottom_up),
                        None,
                        &retina_viewport,
                    )?;
                    flip_rgb_rows(
                        &self.eye_raw_bottom_up,
                        &mut self.eye_raw_top_down,
                        RETINA_WIDTH,
                        RETINA_HEIGHT,
                    );
                    if options.show_eye_view {
                        let retina_top_down =
                            self.retinas[eye_index].process_top_down(&self.eye_raw_top_down)?;
                        flip_rgb_rows(
                            retina_top_down,
                            &mut self.eye_display_bottom_up[eye_index],
                            RETINA_WIDTH,
                            RETINA_HEIGHT,
                        );
                    } else {
                        self.retinas[eye_index].sample_top_down(&self.eye_raw_top_down)?;
                    }
                    self.retina_summaries[eye_index] = self.retinas[eye_index].summary();
                }
                if options.show_eye_view {
                    let eye_viewports = retina_inset_rects(width, height);
                    let source_viewport =
                        MjrRectangle::new(0, 0, RETINA_WIDTH as c_int, RETINA_HEIGHT as c_int);
                    context.offscreen();
                    for (eye_index, eye_viewport) in eye_viewports.iter().enumerate() {
                        unsafe {
                            mjr_drawPixels(
                                self.eye_display_bottom_up[eye_index].as_ptr(),
                                std::ptr::null(),
                                source_viewport,
                                context.ffi(),
                            );
                            mjr_blitBuffer(source_viewport, *eye_viewport, 1, 0, context.ffi());
                        }
                    }
                    context.window();
                    for (eye_index, eye_viewport) in eye_viewports.into_iter().enumerate() {
                        context.overlay(
                            MjtFont::mjFONT_NORMAL,
                            MjtGridPos::mjGRID_TOPLEFT,
                            eye_viewport,
                            if eye_index == 0 {
                                "LEFT | 721 OMMATIDIAL SAMPLES"
                            } else {
                                "RIGHT | 721 OMMATIDIAL SAMPLES"
                            },
                            None,
                        );
                    }
                } else {
                    context.window();
                }
            }
            if options.show_brain_graph {
                let graph_width = (width as f64 * 0.38).round() as c_int;
                let graph_height = (height as f64 * 0.28).round() as c_int;
                let graph_viewport =
                    MjrRectangle::new(width - graph_width - 16, 16, graph_width, graph_height);
                self.brain_figure.draw(graph_viewport, context);
            }
            unsafe { glfwSwapBuffers(self.window.as_ptr()) };
        }
        Ok(())
    }

    pub fn set_title(&self, title: &str) -> Result<()> {
        let title = CString::new(title)?;
        unsafe { glfwSetWindowTitle(self.window.as_ptr(), title.as_ptr()) };
        Ok(())
    }

    pub fn record_brain_field_sample(
        &mut self,
        time_seconds: f64,
        field_potential_mv: f64,
        dominant_frequency_hz: f64,
    ) {
        if self
            .last_brain_sample_time
            .is_some_and(|previous| time_seconds < previous)
        {
            self.clear_brain_history();
        } else if self
            .last_brain_sample_time
            .is_some_and(|previous| time_seconds == previous)
        {
            return;
        }
        self.last_brain_sample_time = Some(time_seconds);
        self.brain_samples
            .push_back((time_seconds as f32, field_potential_mv as f32));
        while self
            .brain_samples
            .front()
            .is_some_and(|(time, _)| *time < time_seconds as f32 - 10.0)
        {
            self.brain_samples.pop_front();
        }
        self.brain_dominant_frequency_hz = dominant_frequency_hz;
        self.brain_figure_dirty = true;
    }

    fn refresh_brain_figure(&mut self) {
        if !self.brain_figure_dirty {
            return;
        }
        self.brain_figure.clear(None);
        for &(time, field) in &self.brain_samples {
            self.brain_figure.push(0, time, field).unwrap();
        }
        let time_seconds = self.last_brain_sample_time.unwrap_or(0.0) as f32;
        let min_time = (time_seconds - 10.0).max(0.0);
        let max_time = time_seconds.max(min_time + 1.0);
        let max_abs_field = self
            .brain_samples
            .iter()
            .map(|(_, field)| field.abs())
            .fold(0.001_f32, f32::max);
        self.brain_figure.range = [
            [min_time, max_time],
            [-max_abs_field * 1.15, max_abs_field * 1.15],
        ];
        self.brain_figure.set_title(&format!(
            "EEG-like field potential | NETWORK PROXY | dominant {:4.1} Hz",
            self.brain_dominant_frequency_hz
        ));
        self.brain_figure_dirty = false;
    }

    pub fn clear_brain_history(&mut self) {
        self.brain_samples.clear();
        self.brain_figure.clear(None);
        self.last_brain_sample_time = None;
        self.brain_figure_dirty = false;
        self.brain_dominant_frequency_hz = 0.0;
    }

    pub fn retina_summaries(&self) -> [RetinaSummary; 2] {
        self.retina_summaries
    }

    fn new_chase_camera(model: &MjModel) -> Result<MjvCamera> {
        Self::new_tracking_camera(
            model,
            CHASE_DISTANCE,
            CHASE_AZIMUTH_DEG,
            CHASE_ELEVATION_DEG,
        )
    }

    fn new_tracking_camera(
        model: &MjModel,
        distance: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
    ) -> Result<MjvCamera> {
        let body_id = model
            .name_to_id(MjtObj::mjOBJ_BODY, CHASE_BODY_NAME)
            .with_context(|| format!("model is missing chase camera body {CHASE_BODY_NAME}"))?;
        let mut camera = MjvCamera::new_tracking(body_id);
        camera.distance = distance;
        camera.azimuth = azimuth_deg;
        camera.elevation = elevation_deg;
        Ok(camera)
    }

    fn use_chase_camera(&mut self, model: &MjModel) {
        if let Ok(camera) = Self::new_chase_camera(model) {
            self.camera = camera;
            self.tracking_camera = Some(TrackingCameraProfile::new(
                CHASE_DISTANCE,
                CHASE_AZIMUTH_DEG,
                CHASE_ELEVATION_DEG,
            ));
            self.hide_fly_in_main_view = false;
        }
    }

    fn use_tracking_camera(
        &mut self,
        model: &MjModel,
        distance: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
    ) {
        if let Ok(camera) = Self::new_tracking_camera(model, distance, azimuth_deg, elevation_deg) {
            self.camera = camera;
            self.tracking_camera = Some(TrackingCameraProfile::new(
                distance,
                azimuth_deg,
                elevation_deg,
            ));
            self.hide_fly_in_main_view = false;
        }
    }

    fn use_fixed_camera(&mut self, model: &MjModel, name: &str, hide_fly: bool) {
        if let Some(camera_id) = model.name_to_id(MjtObj::mjOBJ_CAMERA, name) {
            self.camera.fix(camera_id);
            self.tracking_camera = None;
            self.hide_fly_in_main_view = hide_fly;
        }
    }

    fn key_down(&self, key: c_int) -> bool {
        unsafe { glfwGetKey(self.window.as_ptr(), key) == GLFW_PRESS }
    }

    fn key_pressed(&mut self, key: c_int) -> bool {
        let down = self.key_down(key);
        let was_down = self.pressed_keys.contains(&key);
        if down {
            self.pressed_keys.insert(key);
        } else {
            self.pressed_keys.remove(&key);
        }
        down && !was_down
    }

    fn read_cursor(&mut self) {
        unsafe {
            glfwGetCursorPos(
                self.window.as_ptr(),
                &mut self.cursor[0],
                &mut self.cursor[1],
            )
        };
        self.cursor_initialized = true;
    }

    fn update_camera_from_mouse(&mut self, model: &MjModel) {
        let mut next = [0.0; 2];
        unsafe { glfwGetCursorPos(self.window.as_ptr(), &mut next[0], &mut next[1]) };
        if !self.cursor_initialized {
            self.cursor = next;
            self.cursor_initialized = true;
            return;
        }
        let delta = [next[0] - self.cursor[0], next[1] - self.cursor[1]];
        self.cursor = next;
        let action = if unsafe { glfwGetMouseButton(self.window.as_ptr(), MOUSE_LEFT) }
            == GLFW_PRESS
        {
            Some(MjtMouse::mjMOUSE_ROTATE_V)
        } else if unsafe { glfwGetMouseButton(self.window.as_ptr(), MOUSE_RIGHT) } == GLFW_PRESS {
            Some(MjtMouse::mjMOUSE_MOVE_V)
        } else if unsafe { glfwGetMouseButton(self.window.as_ptr(), MOUSE_MIDDLE) } == GLFW_PRESS {
            Some(MjtMouse::mjMOUSE_ZOOM)
        } else {
            None
        };
        if let Some(action) = action
            && let Some(scene) = self.scene.as_ref()
        {
            self.camera
                .move_(action, model, delta[0] / 600.0, delta[1] / 600.0, scene);
            if let Some(profile) = self.tracking_camera.as_mut() {
                profile.nominal_distance = self.camera.distance.max(0.0);
                profile.nominal_azimuth_deg = self.camera.azimuth;
                profile.elevation_deg = self.camera.elevation;
                profile.azimuth_offset_deg = 0.0;
            }
        }
    }

    fn hud_button_clicks(&mut self) -> (bool, bool, bool) {
        let left_down =
            unsafe { glfwGetMouseButton(self.window.as_ptr(), MOUSE_LEFT) } == GLFW_PRESS;
        let clicked = left_down && !self.mouse_left_was_down;
        self.mouse_left_was_down = left_down;
        if !clicked {
            return (false, false, false);
        }
        let mut framebuffer_width = 0;
        let mut framebuffer_height = 0;
        let mut window_width = 0;
        let mut window_height = 0;
        unsafe {
            glfwGetFramebufferSize(
                self.window.as_ptr(),
                &mut framebuffer_width,
                &mut framebuffer_height,
            );
            glfwGetWindowSize(self.window.as_ptr(), &mut window_width, &mut window_height);
        }
        if framebuffer_width <= 0
            || framebuffer_height <= 0
            || window_width <= 0
            || window_height <= 0
        {
            return (false, false, false);
        }
        let cursor_x = self.cursor[0] * f64::from(framebuffer_width) / f64::from(window_width);
        let cursor_y = f64::from(framebuffer_height)
            - self.cursor[1] * f64::from(framebuffer_height) / f64::from(window_height);
        let [eye_button, brain_button, flight_button] =
            hud_button_rects(framebuffer_width, framebuffer_height);
        (
            rectangle_contains(eye_button, cursor_x, cursor_y),
            rectangle_contains(brain_button, cursor_x, cursor_y),
            rectangle_contains(flight_button, cursor_x, cursor_y),
        )
    }
}

impl Drop for LiveViewer {
    fn drop(&mut self) {
        unsafe { glfwMakeContextCurrent(self.window.as_ptr()) };
        self.scene.take();
        self.eye_scene.take();
        self.context.take();
        unsafe {
            glfwMakeContextCurrent(std::ptr::null_mut());
            glfwDestroyWindow(self.window.as_ptr());
            glfwTerminate();
        }
    }
}

fn update_collision_aware_tracking_scene<M>(
    scene: &mut MjvScene,
    data: &mut MjData<M>,
    option: &MjvOption,
    perturb: &MjvPerturb,
    camera: &mut MjvCamera,
    profile: &mut TrackingCameraProfile,
) where
    M: std::ops::Deref<Target = MjModel>,
{
    let nominal_safe =
        tracking_candidate_safe_distance(scene, data, option, perturb, camera, profile, 0.0);
    let keep_offset =
        profile.azimuth_offset_deg != 0.0 && nominal_safe < TRACKING_CAMERA_RESTORE_DISTANCE;
    let mut selected_offset = if keep_offset {
        profile.azimuth_offset_deg
    } else {
        0.0
    };
    let mut selected_safe = tracking_candidate_safe_distance(
        scene,
        data,
        option,
        perturb,
        camera,
        profile,
        selected_offset,
    );
    if selected_safe < TRACKING_CAMERA_MIN_DISTANCE {
        for offset in TRACKING_CAMERA_AZIMUTH_OFFSETS {
            let safe = tracking_candidate_safe_distance(
                scene, data, option, perturb, camera, profile, offset,
            );
            if safe > selected_safe {
                selected_offset = offset;
                selected_safe = safe;
            }
        }
    }
    profile.azimuth_offset_deg = selected_offset;
    camera.azimuth = profile.nominal_azimuth_deg + selected_offset;
    camera.elevation = profile.elevation_deg;
    camera.distance = clamped_tracking_distance(profile.nominal_distance, selected_safe);
    scene.update(data, option, perturb, camera);
}

fn tracking_candidate_safe_distance<M>(
    scene: &mut MjvScene,
    data: &mut MjData<M>,
    option: &MjvOption,
    perturb: &MjvPerturb,
    camera: &mut MjvCamera,
    profile: &TrackingCameraProfile,
    azimuth_offset_deg: f64,
) -> f64
where
    M: std::ops::Deref<Target = MjModel>,
{
    camera.azimuth = profile.nominal_azimuth_deg + azimuth_offset_deg;
    camera.elevation = profile.elevation_deg;
    camera.distance = profile.nominal_distance;
    scene.update(data, option, perturb, camera);
    let gl_cameras = &scene.ffi().camera;
    let camera_position = std::array::from_fn(|axis| {
        0.5 * f64::from(gl_cameras[0].pos[axis] + gl_cameras[1].pos[axis])
    });
    safe_tracking_distance(camera.lookat, camera_position, TRACKING_CAMERA_WALL_BOUNDS)
}

fn safe_tracking_distance(
    target: [f64; 3],
    camera_position: [f64; 3],
    bounds: [[f64; 2]; 3],
) -> f64 {
    let offset = std::array::from_fn::<_, 3, _>(|axis| camera_position[axis] - target[axis]);
    let norm = offset.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= 1e-9 {
        return 0.0;
    }
    let direction = offset.map(|value| value / norm);
    let mut safe_distance = f64::INFINITY;
    for axis in 0..3 {
        let axis_distance = if direction[axis] > 1e-9 {
            (bounds[axis][1] - target[axis]) / direction[axis]
        } else if direction[axis] < -1e-9 {
            (bounds[axis][0] - target[axis]) / direction[axis]
        } else {
            f64::INFINITY
        };
        safe_distance = safe_distance.min(axis_distance);
    }
    safe_distance.max(0.0)
}

fn clamped_tracking_distance(nominal_distance: f64, safe_distance: f64) -> f64 {
    nominal_distance.min((safe_distance - 1.0).max(0.0))
}

fn move_food_geom(
    scene: &mut MjvScene,
    food_geom_id: usize,
    food_center: [f64; 3],
    food_enabled: bool,
) {
    for geom in unsafe { scene.geoms_mut() } {
        if geom.objtype == MjtObj::mjOBJ_GEOM as c_int && geom.objid == food_geom_id as c_int {
            geom.pos = food_center.map(|value| value as f32);
            geom.rgba = if food_enabled {
                [0.28, 0.85, 0.20, 1.0]
            } else {
                [0.35, 0.35, 0.35, 0.25]
            };
            break;
        }
    }
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "ON" } else { "OFF" }
}

fn hud_button_rects(width: c_int, height: c_int) -> [MjrRectangle; 3] {
    let button_width = 180;
    let gap = 12;
    let left = (width - button_width * 3 - gap * 2) / 2;
    [
        MjrRectangle::new(left, height - 48, button_width, 32),
        MjrRectangle::new(left + button_width + gap, height - 48, button_width, 32),
        MjrRectangle::new(
            left + (button_width + gap) * 2,
            height - 48,
            button_width,
            32,
        ),
    ]
}

fn retina_inset_rects(width: c_int, height: c_int) -> [MjrRectangle; 2] {
    let margin = 16;
    let gap = 8;
    let combined_width = (width as f64 * 0.36).round() as c_int;
    let inset_width = (combined_width - gap) / 2;
    let inset_height =
        ((f64::from(inset_width) * RETINA_HEIGHT as f64) / RETINA_WIDTH as f64).round() as c_int;
    let left = width - combined_width - margin;
    let bottom = height - inset_height - margin;
    [
        MjrRectangle::new(left, bottom, inset_width, inset_height),
        MjrRectangle::new(left + inset_width + gap, bottom, inset_width, inset_height),
    ]
}

fn hide_fly_visuals(scene: &mut MjvScene, model: &MjModel) {
    for geom in unsafe { scene.geoms_mut() } {
        let object_type = if geom.objtype == MjtObj::mjOBJ_GEOM as c_int {
            Some(MjtObj::mjOBJ_GEOM)
        } else if geom.objtype == MjtObj::mjOBJ_SITE as c_int {
            Some(MjtObj::mjOBJ_SITE)
        } else {
            None
        };
        if object_type.is_some_and(|object_type| {
            usize::try_from(geom.objid)
                .ok()
                .and_then(|id| model.id_to_name(object_type, id))
                .is_some_and(|name| name.starts_with("fly/"))
        }) {
            geom.type_ = MjtGeom::mjGEOM_NONE as c_int;
        }
    }
}

fn flip_rgb_rows(source: &[u8], destination: &mut [u8], width: usize, height: usize) {
    let row_bytes = width * 3;
    debug_assert_eq!(source.len(), row_bytes * height);
    debug_assert_eq!(destination.len(), row_bytes * height);
    for row in 0..height {
        let source_start = row * row_bytes;
        let destination_start = (height - row - 1) * row_bytes;
        destination[destination_start..destination_start + row_bytes]
            .copy_from_slice(&source[source_start..source_start + row_bytes]);
    }
}

fn rectangle_contains(rectangle: MjrRectangle, x: f64, y: f64) -> bool {
    x >= f64::from(rectangle.left)
        && x < f64::from(rectangle.left + rectangle.width)
        && y >= f64::from(rectangle.bottom)
        && y < f64::from(rectangle.bottom + rectangle.height)
}

fn draw_hud_button(context: &MjrContext, rectangle: MjrRectangle, label: &str, enabled: bool) {
    let label = CString::new(label).unwrap();
    let background = if enabled {
        [0.05, 0.35, 0.28, 0.92]
    } else {
        [0.12, 0.14, 0.18, 0.88]
    };
    unsafe {
        mjr_label(
            rectangle,
            MjtFont::mjFONT_NORMAL as c_int,
            label.as_ptr(),
            background[0],
            background[1],
            background[2],
            background[3],
            0.92,
            0.96,
            1.0,
            context.ffi(),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::{
        TRACKING_CAMERA_WALL_BOUNDS, clamped_tracking_distance, retina_inset_rects,
        safe_tracking_distance,
    };

    #[test]
    fn binocular_retina_panels_are_equal_and_side_by_side() {
        let [left, right] = retina_inset_rects(1_280, 800);
        assert_eq!(left.width, right.width);
        assert_eq!(left.height, right.height);
        assert!(left.left + left.width < right.left);
        assert!(right.left + right.width <= 1_280);
        assert!(left.bottom + left.height <= 800);
    }

    #[test]
    fn tracking_camera_distance_stays_inside_room_walls() {
        let centered = safe_tracking_distance(
            [0.0, 0.0, 30.0],
            [36.0, 0.0, 40.0],
            TRACKING_CAMERA_WALL_BOUNDS,
        );
        assert!(centered > 36.0);

        let outward = safe_tracking_distance(
            [296.0, 0.0, 30.0],
            [332.0, 0.0, 40.0],
            TRACKING_CAMERA_WALL_BOUNDS,
        );
        let inward = safe_tracking_distance(
            [296.0, 0.0, 30.0],
            [260.0, 0.0, 40.0],
            TRACKING_CAMERA_WALL_BOUNDS,
        );
        assert!(outward < 2.0);
        assert!(inward > 36.0);

        let corner_outward = safe_tracking_distance(
            [296.0, 216.0, 30.0],
            [320.0, 240.0, 40.0],
            TRACKING_CAMERA_WALL_BOUNDS,
        );
        assert!(corner_outward < 2.0);
    }

    #[test]
    fn tracking_camera_distance_never_exceeds_wall_clearance() {
        assert_eq!(clamped_tracking_distance(36.0, 0.25), 0.0);
        assert_eq!(clamped_tracking_distance(36.0, 1.5), 0.5);
        assert_eq!(clamped_tracking_distance(36.0, 50.0), 36.0);
    }
}
