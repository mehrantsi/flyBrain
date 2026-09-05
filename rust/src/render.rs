use std::fs;
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use anyhow::{Context, Result, bail};
use mujoco_rs::prelude::*;
use mujoco_rs::renderer::MjRenderer;

pub struct FlyRenderer {
    renderer: MjRenderer,
    presentation: RoomPresentation,
    eye_camera: bool,
    width: u32,
    height: u32,
}

impl FlyRenderer {
    pub fn fixed_camera(
        model: &MjModel,
        camera_name: &str,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("render dimensions must be positive")
        }
        let camera_id = model
            .name_to_id(MjtObj::mjOBJ_CAMERA, camera_name)
            .with_context(|| format!("model has no camera named {camera_name}"))?;
        let renderer = MjRenderer::builder()
            .width(width)
            .height(height)
            .rgb(true)
            .depth(false)
            .camera(MjvCamera::new_fixed(camera_id))
            .build(model)
            .context("initializing the MuJoCo offscreen renderer")?;
        Ok(Self {
            renderer,
            presentation: RoomPresentation::new(model),
            eye_camera: camera_name.ends_with("eye_cam_camera"),
            width,
            height,
        })
    }

    pub fn render<M>(&mut self, data: &mut MjData<M>) -> Result<&[u8]>
    where
        M: Deref<Target = MjModel>,
    {
        self.renderer
            .sync_data(data)
            .context("syncing MuJoCo render state")?;
        if !self.eye_camera {
            self.presentation.apply(self.renderer.scene_mut());
        }
        self.renderer.render().context("rendering MuJoCo frame")?;
        self.renderer
            .rgb_flat()
            .context("RGB rendering is disabled")
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        self.renderer
            .save_rgb(path)
            .with_context(|| format!("saving {}", path.display()))
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

pub struct RoomPresentation {
    walls: Vec<(i32, usize, f32, f32)>,
    fly_geom_ids: Vec<i32>,
}

impl RoomPresentation {
    pub fn new(model: &MjModel) -> Self {
        let mut walls: Vec<_> = [
            ("room_wall_left", 0, -1.0),
            ("room_wall_right", 0, 1.0),
            ("room_wall_back", 1, 1.0),
            ("room_wall_front_left", 1, -1.0),
            ("room_wall_front_right", 1, -1.0),
            ("room_wall_front_window", 1, -1.0),
            ("room_wall_ceiling", 2, 1.0),
        ]
        .into_iter()
        .filter_map(|(name, axis, sign)| {
            let id = model.name_to_id(MjtObj::mjOBJ_GEOM, name)?;
            let inner_face =
                model.geom_pos()[id][axis] as f32 * sign - model.geom_size()[id][axis] as f32;
            Some((id as i32, axis, sign, inner_face))
        })
        .collect();
        for (wall, prefixes, axis, sign) in [
            (
                "room_wall_right",
                ["window_", "detail_baseboard_right"],
                0,
                1.0,
            ),
            (
                "room_wall_left",
                ["detail_baseboard_left", "detail_wall_left_"],
                0,
                -1.0,
            ),
            (
                "room_wall_back",
                ["detail_baseboard_back", "detail_wall_panel_"],
                1,
                1.0,
            ),
            (
                "room_wall_front_left",
                ["detail_baseboard_front", "detail_wall_front_"],
                1,
                -1.0,
            ),
        ] {
            let Some(wall_id) = model.name_to_id(MjtObj::mjOBJ_GEOM, wall) else {
                continue;
            };
            let inner_face = model.geom_pos()[wall_id][axis] as f32 * sign
                - model.geom_size()[wall_id][axis] as f32;
            for id in 0..model.ngeom() as usize {
                if model
                    .id_to_name(MjtObj::mjOBJ_GEOM, id)
                    .is_some_and(|name| prefixes.iter().any(|prefix| name.starts_with(prefix)))
                {
                    walls.push((id as i32, axis, sign, inner_face));
                }
            }
        }
        Self {
            walls,
            fly_geom_ids: model
                .name_to_id(MjtObj::mjOBJ_BODY, "fly/c_thorax")
                .map(|body| {
                    model
                        .geom_bodyid()
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| **id as usize == body)
                        .map(|(id, _)| id as i32)
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    pub fn apply(&self, scene: &mut MjvScene) {
        let position: [f32; 3] = std::array::from_fn(|axis| {
            0.5 * (scene.camera()[0].pos[axis] + scene.camera()[1].pos[axis])
        });
        let distance = scene
            .geoms()
            .iter()
            .find(|geom| {
                geom.objtype == MjtObj::mjOBJ_GEOM as i32 && self.fly_geom_ids.contains(&geom.objid)
            })
            .map(|geom| {
                position
                    .iter()
                    .zip(geom.pos)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .unwrap_or(100.0);
        for camera in scene.camera_mut() {
            set_near_plane(camera, (0.02 * distance).clamp(0.05, 2.0));
        }
        for geom in unsafe { scene.geoms_mut() } {
            if geom.objtype == MjtObj::mjOBJ_GEOM as i32
                && self.walls.iter().any(|&(id, axis, sign, inner_face)| {
                    geom.objid == id && position[axis] * sign > inner_face
                })
            {
                geom.type_ = MjtGeom::mjGEOM_NONE as i32;
            }
        }
    }
}

fn set_near_plane(camera: &mut MjvGLCamera, near: f32) {
    if camera.orthographic == 0 {
        let scale = near / camera.frustum_near;
        camera.frustum_center *= scale;
        camera.frustum_width *= scale;
        camera.frustum_bottom *= scale;
        camera.frustum_top *= scale;
    }
    camera.frustum_near = near;
}

pub struct VideoRecorder {
    child: Child,
    input: Option<ChildStdin>,
    output: PathBuf,
    expected_frame_bytes: usize,
    frame_count: u64,
}

impl VideoRecorder {
    pub fn new(
        output: impl AsRef<Path>,
        width: u32,
        height: u32,
        fps: u32,
        overwrite: bool,
    ) -> Result<Self> {
        if width == 0 || height == 0 || fps == 0 {
            bail!("video width, height, and fps must be positive")
        }
        let output = output.as_ref().to_path_buf();
        if output.exists() && !overwrite {
            bail!(
                "video already exists: {}; pass --force to replace it",
                output.display()
            )
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let replace_flag = if overwrite { "-y" } else { "-n" };
        let dimensions = format!("{width}x{height}");
        let frame_rate = fps.to_string();
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                replace_flag,
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &dimensions,
                "-framerate",
                &frame_rate,
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(&output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting ffmpeg; install ffmpeg to encode MP4 output")?;
        let input = child.stdin.take().context("opening ffmpeg input pipe")?;
        let expected_frame_bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("video dimensions overflow")?;
        Ok(Self {
            child,
            input: Some(input),
            output,
            expected_frame_bytes,
            frame_count: 0,
        })
    }

    pub fn write_frame(&mut self, rgb: &[u8]) -> Result<()> {
        if rgb.len() != self.expected_frame_bytes {
            bail!(
                "RGB frame has {} bytes, expected {}",
                rgb.len(),
                self.expected_frame_bytes
            )
        }
        self.input
            .as_mut()
            .context("video recorder is already finished")?
            .write_all(rgb)
            .context("writing RGB frame to ffmpeg")?;
        self.frame_count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<VideoSummary> {
        self.input.take();
        let output = self
            .child
            .wait_with_output()
            .context("waiting for ffmpeg")?;
        if !output.status.success() {
            bail!(
                "ffmpeg failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
        }
        let bytes = fs::metadata(&self.output)
            .with_context(|| format!("reading {} metadata", self.output.display()))?
            .len();
        Ok(VideoSummary {
            path: self.output,
            frames: self.frame_count,
            bytes,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoSummary {
    pub path: PathBuf,
    pub frames: u64,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_cutaway_preserves_projection_physics_and_subsequent_eye_scenes() {
        let model = MjModel::from_xml("assets/neuromechfly/fly.xml").unwrap();
        let mut data = model.make_data();
        data.forward();
        let state = data.qpos().to_vec();
        let mut scene = MjvScene::new(&model, model.ngeom() as usize + 64);
        let mut camera = MjvCamera::new_fixed(
            model
                .name_to_id(MjtObj::mjOBJ_CAMERA, "room_camera")
                .unwrap(),
        );
        scene.update(
            &mut data,
            &MjvOption::default(),
            &MjvPerturb::default(),
            &mut camera,
        );
        let original = scene.camera().clone();
        RoomPresentation::new(&model).apply(&mut scene);
        for (before, after) in original.iter().zip(scene.camera()) {
            assert!(after.frustum_near / before.frustum_near > 50.0);
            for (a, b) in [
                (before.frustum_top, after.frustum_top),
                (before.frustum_bottom, after.frustum_bottom),
                (before.frustum_center, after.frustum_center),
                (before.frustum_width, after.frustum_width),
            ] {
                assert!((a / before.frustum_near - b / after.frustum_near).abs() < 1e-5);
            }
            assert_eq!(before.frustum_far, after.frustum_far);
        }
        let roof = model
            .name_to_id(MjtObj::mjOBJ_GEOM, "room_wall_ceiling")
            .unwrap() as i32;
        assert!(
            scene
                .geoms()
                .iter()
                .any(|geom| geom.objid == roof && geom.type_ == MjtGeom::mjGEOM_NONE as i32)
        );
        assert_eq!(model.geom_conaffinity()[roof as usize], 1);
        assert_eq!(data.qpos(), state);
        for name in ["fly/l_eye_cam_camera", "fly/r_eye_cam_camera"] {
            let mut camera =
                MjvCamera::new_fixed(model.name_to_id(MjtObj::mjOBJ_CAMERA, name).unwrap());
            scene.update(
                &mut data,
                &MjvOption::default(),
                &MjvPerturb::default(),
                &mut camera,
            );
            assert!(scene.geoms().iter().any(|geom| geom.objid == roof
                && geom.type_ == MjtGeom::mjGEOM_BOX as i32
                && geom.rgba[3] == 1.0));
            assert!((scene.camera()[0].frustum_near - 0.035).abs() < 1e-6);
        }
    }

    #[test]
    fn recorder_rejects_bad_dimensions() {
        assert!(VideoRecorder::new("ignored.mp4", 0, 1, 30, false).is_err());
    }
}
