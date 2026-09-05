# Habitat and fly presentation

The appearance pass adds an oak floor and furniture surfaces, rounded furniture,
rimmed dishes, a hollow cup, shaped fruit, plant stems and veined leaves, wall
trim, and anatomical fly detail (bristles, abdominal bands, and cosmetic eye
facets). The oak bitmap was generated with ImageGen; its prompt and provenance
are recorded in `assets/materials/README.md`. The remaining detail geometry is
generated deterministically by the repository's Python asset tools.

Original fly and habitat collision geometry is retained. New cosmetic geometry
has zero mass, no collision masks, and no fluid forces. Appearance refresh does
not change resource locations, odor concentrations, motor calibration, or the
connectome. It does change visible sensory input: a more detailed scene is not
a claim of behavioral equivalence to the former textures and lighting.

The native observer uses an adaptive near clipping plane, rescales the frustum
to retain the field of view, and hides intervening room walls only in the
observer scene. The two sensory-eye scenes retain their original field of view
and closed room geometry. The floor's planar reflection and overlapping
transparent room shell were removed; antialiasing and valid room lighting are
enabled. These changes address depth fighting, ghost reflections, and the dark
upper-wall band seen in the former presentation.

Refresh the generated appearance and browser manifest together:

```bash
.venv/bin/python tools/refresh_appearance.py
.venv/bin/python tools/package_browser_assets.py
```

Generate an eight-view native gallery in a new directory:

```bash
cargo run --locked --release --example room_preview -- outputs/world/my-gallery
```

The gallery warms the renderer, then compares eight repeated static frames per
view. This is a reproducible static stability check, not a guarantee against
every moving-camera transparency artifact. MuJoCo's classic OpenGL renderer is
not a photorealistic path tracer. Browser materials use Three.js and are not
pixel-equivalent to native.

Eye facets on the fly mesh are cosmetic. The binocular HUD still displays the
FlyGym 721-ommatidium sampling map; it does not imply that a fly experiences a
literal screen of hexagonal tiles. The field trace is a simulated network
potential proxy, not a biological EEG recording.
