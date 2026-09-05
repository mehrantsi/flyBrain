use anyhow::{Context, Result, ensure};
use flybrain_engine::render::RoomPresentation;
use mujoco_rs::prelude::*;
use mujoco_rs::renderer::MjRenderer;
use std::path::PathBuf;

fn main() -> Result<()> {
    let output = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("provide a new output directory")?,
    );
    ensure!(!output.exists(), "output directory already exists");
    std::fs::create_dir_all(&output)?;
    let model = MjModel::from_xml("assets/neuromechfly/fly.xml")?;
    let mut data = model.make_data();
    data.reset_keyframe(0)?;
    data.forward();
    let presentation = RoomPresentation::new(&model);
    let mut renderer = MjRenderer::builder()
        .width(960)
        .height(720)
        .rgb(true)
        .depth(false)
        .build(&model)?;
    let fixed = |name| MjvCamera::new_fixed(model.name_to_id(MjtObj::mjOBJ_CAMERA, name).unwrap());
    let free = |lookat, distance, azimuth, elevation| {
        let mut camera = MjvCamera::new_free(&model);
        camera.lookat = lookat;
        camera.distance = distance;
        camera.azimuth = azimuth;
        camera.elevation = elevation;
        camera
    };
    for (name, camera, eye) in [
        ("room", fixed("room_camera"), false),
        ("fly", free([-0.4, 0.0, 2.1], 4.5, 125.0, -22.0), false),
        ("food", free([26.0, 13.0, 2.0], 48.0, 135.0, -26.0), false),
        ("water", free([-58.0, -34.0, 0.5], 30.0, 90.0, -55.0), false),
        (
            "table",
            free([127.0, 70.0, 57.0], 85.0, 125.0, -24.0),
            false,
        ),
        (
            "plant",
            free([220.0, -130.0, 25.0], 80.0, 145.0, -22.0),
            false,
        ),
        ("left-eye", fixed("fly/l_eye_cam_camera"), true),
        ("right-eye", fixed("fly/r_eye_cam_camera"), true),
    ] {
        renderer.set_camera(camera);
        renderer.sync_data(&mut data)?;
        if eye {
            for geom in unsafe { renderer.scene_mut().geoms_mut() } {
                if geom.objtype == MjtObj::mjOBJ_GEOM as i32
                    && model
                        .id_to_name(MjtObj::mjOBJ_GEOM, geom.objid as usize)
                        .is_some_and(|name| name.starts_with("fly/"))
                {
                    geom.type_ = MjtGeom::mjGEOM_NONE as i32;
                }
            }
        } else {
            presentation.apply(renderer.scene_mut());
        }
        for _ in 0..8 {
            renderer.render()?;
        }
        let first = renderer.rgb_flat().context("RGB buffer")?.to_vec();
        renderer.save_rgb(output.join(format!("{name}.png")))?;
        for _ in 0..8 {
            renderer.render()?;
            let rgb = renderer.rgb_flat().context("RGB buffer")?;
            let changed = rgb.iter().zip(&first).filter(|(a, b)| a != b).count();
            let max_delta = rgb
                .iter()
                .zip(&first)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            ensure!(
                changed == 0,
                "{name}: {changed} channels changed in a static scene (maximum delta {max_delta})"
            );
        }
        println!("{name}: 8 repeated frames identical");
    }
    Ok(())
}
