# Presentation materials

`oak-v1.png` is an AI-generated oak color texture, created with the built-in image-generation tool on 2026-09-05. It is not a photograph or a scientific measurement. The exporter copies it to `assets/neuromechfly/textures/`.

Generation prompt:

> Use case: photorealistic-natural. Asset type: seamless tileable albedo texture for a real-time 3D room floor and wooden furniture. Generate a square high-resolution orthographic surface scan of warm natural European oak boards, matte lightly oiled finish, fine realistic longitudinal grain, subtle pores and occasional small knots, restrained honey and medium-brown tones. Four long staggered boards filling the entire image, narrow hairline seams, visually seamless at all four edges. Uniform flat diffuse lighting, no cast shadows, no perspective, no vignette, no highlights baked into the texture, no objects, text, border or watermark. This is a material color map, not a room picture. Fine material detail visible at close inspection.

Regenerate procedural room and fly details and refresh the asset hashes with:

```sh
.venv/bin/python tools/refresh_appearance.py
```

Cosmetic meshes are separate from the original collision proxies. These details are artistic approximations; they do not add biological fidelity to the neural model or change the compound-eye sampling layout.
