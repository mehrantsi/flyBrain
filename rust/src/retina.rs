use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

pub const RETINA_HEIGHT: usize = 512;
pub const RETINA_WIDTH: usize = 450;
pub const OMMATIDIA_PER_EYE: usize = 721;

const PIXELS: usize = RETINA_HEIGHT * RETINA_WIDTH;
const RGB_BYTES: usize = PIXELS * 3;
const DISTORTION_COEFFICIENT: f64 = 3.8;
const FISHEYE_ZOOM: f64 = 2.72;
const EXPECTED_COVERED_PIXELS: usize = 170_288;
const EXPECTED_PALE_OMMATIDIA: usize = 216;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RetinaSummary {
    pub mean_intensity: f64,
    pub spatial_contrast: f64,
}

pub struct FlyGymRetina {
    ommatidia_id_map: Box<[u16]>,
    pale_mask: Box<[bool]>,
    pixels_per_ommatidium: Box<[u32]>,
    samples: Box<[RetinaSample]>,
    readings: Box<[[f64; 2]]>,
    display_rgb: Box<[u8]>,
}

struct RetinaSample {
    source_byte: u32,
    ommatidium: u16,
}

impl FlyGymRetina {
    pub fn load(assets_dir: impl AsRef<Path>) -> Result<Self> {
        let vision_dir = assets_dir.as_ref().join("vision");
        let map_path = vision_dir.join("ommatidia_id_map_u16le.bin");
        let mask_path = vision_dir.join("pale_mask_u8.bin");
        let map_bytes = fs::read(&map_path)
            .with_context(|| format!("reading FlyGym retina map {}", map_path.display()))?;
        let mask_bytes = fs::read(&mask_path)
            .with_context(|| format!("reading FlyGym pale mask {}", mask_path.display()))?;
        if map_bytes.len() != PIXELS * 2 {
            bail!(
                "FlyGym retina map has {} bytes, expected {}",
                map_bytes.len(),
                PIXELS * 2
            )
        }
        if mask_bytes.len() != OMMATIDIA_PER_EYE {
            bail!(
                "FlyGym pale mask has {} bytes, expected {}",
                mask_bytes.len(),
                OMMATIDIA_PER_EYE
            )
        }
        let id_map = map_bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        let pale_mask = mask_bytes
            .into_iter()
            .map(|value| match value {
                0 => Ok(false),
                1 => Ok(true),
                _ => bail!("FlyGym pale mask contains value {value}; expected 0 or 1"),
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_parts(id_map, pale_mask)
    }

    fn from_parts(ommatidia_id_map: Vec<u16>, pale_mask: Vec<bool>) -> Result<Self> {
        if ommatidia_id_map.len() != PIXELS || pale_mask.len() != OMMATIDIA_PER_EYE {
            bail!("FlyGym retina arrays have invalid shapes")
        }
        let mut pixels_per_ommatidium = vec![0_u32; OMMATIDIA_PER_EYE];
        for &id in &ommatidia_id_map {
            if usize::from(id) > OMMATIDIA_PER_EYE {
                bail!("FlyGym retina map contains out-of-range ommatidium ID {id}")
            }
            if id != 0 {
                pixels_per_ommatidium[usize::from(id - 1)] += 1;
            }
        }
        if pixels_per_ommatidium.contains(&0) {
            bail!("FlyGym retina map does not contain every ommatidium ID")
        }
        if pixels_per_ommatidium
            .iter()
            .map(|&count| count as usize)
            .sum::<usize>()
            != EXPECTED_COVERED_PIXELS
        {
            bail!("FlyGym retina covered-pixel count does not match the pinned asset")
        }
        if pale_mask.iter().filter(|&&pale| pale).count() != EXPECTED_PALE_OMMATIDIA {
            bail!("FlyGym pale-ommatidia count does not match the pinned asset")
        }

        let fisheye = build_fisheye_source_pixels();
        let samples = ommatidia_id_map
            .iter()
            .zip(&fisheye)
            .filter_map(|(&id, &source)| {
                if id == 0 {
                    return None;
                }
                let ommatidium = id - 1;
                source.map(|source| RetinaSample {
                    source_byte: (source * 3 + usize::from(pale_mask[usize::from(ommatidium)]) + 1) as u32,
                    ommatidium,
                })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            ommatidia_id_map: ommatidia_id_map.into_boxed_slice(),
            pale_mask: pale_mask.into_boxed_slice(),
            pixels_per_ommatidium: pixels_per_ommatidium.into_boxed_slice(),
            samples: samples.into_boxed_slice(),
            readings: vec![[0.0; 2]; OMMATIDIA_PER_EYE].into_boxed_slice(),
            display_rgb: vec![0; RGB_BYTES].into_boxed_slice(),
        })
    }

    pub fn process_top_down(&mut self, raw_rgb: &[u8]) -> Result<&[u8]> {
        self.sample_top_down(raw_rgb)?;
        Ok(self.display())
    }

    pub fn display(&mut self) -> &[u8] {
        self.display_rgb.fill(0);
        for (pixel, &id) in self.ommatidia_id_map.iter().enumerate() {
            if id == 0 {
                continue;
            }
            let reading = self.readings[usize::from(id - 1)];
            let intensity = ((reading[0] + reading[1]) * 255.0).clamp(0.0, 255.0) as u8;
            self.display_rgb[pixel * 3..pixel * 3 + 3].fill(intensity);
        }
        &self.display_rgb
    }

    pub fn sample_top_down(&mut self, raw_rgb: &[u8]) -> Result<()> {
        if raw_rgb.len() != RGB_BYTES {
            bail!(
                "raw eye image has {} bytes, expected {}",
                raw_rgb.len(),
                RGB_BYTES
            )
        }
        self.readings.fill([0.0; 2]);
        for sample in &self.samples {
            let ommatidium = usize::from(sample.ommatidium);
            let channel = usize::from(self.pale_mask[ommatidium]);
            self.readings[ommatidium][channel] +=
                f64::from(raw_rgb[sample.source_byte as usize])
                    / f64::from(self.pixels_per_ommatidium[ommatidium]);
        }
        for reading in &mut self.readings {
            reading[0] /= 255.0;
            reading[1] /= 255.0;
        }
        Ok(())
    }

    pub fn readings(&self) -> &[[f64; 2]] {
        &self.readings
    }

    pub fn summary(&self) -> RetinaSummary {
        let mean_intensity = self
            .readings
            .iter()
            .map(|reading| reading[0] + reading[1])
            .sum::<f64>()
            / OMMATIDIA_PER_EYE as f64;
        let variance = self
            .readings
            .iter()
            .map(|reading| {
                let delta = reading[0] + reading[1] - mean_intensity;
                delta * delta
            })
            .sum::<f64>()
            / OMMATIDIA_PER_EYE as f64;
        RetinaSummary {
            mean_intensity,
            spatial_contrast: variance.sqrt(),
        }
    }
}

fn build_fisheye_source_pixels() -> Vec<Option<usize>> {
    let mut sources = Vec::with_capacity(PIXELS);
    let height = RETINA_HEIGHT as f64;
    let width = RETINA_WIDTH as f64;
    for destination_row in 0..RETINA_HEIGHT {
        for destination_column in 0..RETINA_WIDTH {
            let row = ((2.0 * destination_row as f64 - height) / height) / FISHEYE_ZOOM;
            let column = ((2.0 * destination_column as f64 - width) / width) / FISHEYE_ZOOM;
            let denominator = 1.0 - DISTORTION_COEFFICIENT * (column * column + row * row) + 1e-6;
            let source_row = (((row / denominator + 1.0) * height) / 2.0) as isize;
            let source_column = (((column / denominator + 1.0) * width) / 2.0) as isize;
            let source = if (0..RETINA_HEIGHT as isize).contains(&source_row)
                && (0..RETINA_WIDTH as isize).contains(&source_column)
            {
                Some(source_row as usize * RETINA_WIDTH + source_column as usize)
            } else {
                None
            };
            sources.push(source);
        }
    }
    sources
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn assets_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/neuromechfly")
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn loads_pinned_flygym_retina_assets() {
        let retina = FlyGymRetina::load(assets_dir()).expect("load retina");
        assert_eq!(retina.ommatidia_id_map.len(), PIXELS);
        assert_eq!(retina.pale_mask.len(), OMMATIDIA_PER_EYE);
        assert_eq!(
            retina
                .pixels_per_ommatidium
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            EXPECTED_COVERED_PIXELS
        );
        assert_eq!(
            retina.pale_mask.iter().filter(|&&pale| pale).count(),
            EXPECTED_PALE_OMMATIDIA
        );
    }

    #[test]
    fn matches_flygym_reference_transform() {
        let mut raw = vec![0_u8; RGB_BYTES];
        for row in 0..RETINA_HEIGHT {
            for column in 0..RETINA_WIDTH {
                let pixel = (row * RETINA_WIDTH + column) * 3;
                raw[pixel] = ((row * 3 + column * 5) % 256) as u8;
                raw[pixel + 1] = ((row * 7 + column * 11 + 13) % 256) as u8;
                raw[pixel + 2] = ((row * 17 + column * 19 + 29) % 256) as u8;
            }
        }
        let mut retina = FlyGymRetina::load(assets_dir()).expect("load retina");
        let display = retina.process_top_down(&raw).expect("process retina");
        assert_eq!(
            sha256(display),
            "1a8bd29f8c09b13f3e1e8808b01200c743eaa93026b52f305acb4ae0687abd7a"
        );
        assert!((retina.readings()[0][0] - 0.498_495_605_138_607_4).abs() < 1e-14);
        assert!((retina.readings()[360][1] - 0.684_211_423_699_915_1).abs() < 1e-14);
        assert!((retina.readings()[720][0] - 0.510_835_023_664_638_3).abs() < 1e-14);
        let summary = retina.summary();
        assert!(summary.mean_intensity > 0.45 && summary.mean_intensity < 0.75);
        assert!(summary.spatial_contrast > 0.05);
    }

    #[test]
    fn summary_only_path_matches_the_display_path() {
        let raw = vec![127_u8; RGB_BYTES];
        let mut display_retina = FlyGymRetina::load(assets_dir()).unwrap();
        let mut summary_retina = FlyGymRetina::load(assets_dir()).unwrap();
        display_retina.process_top_down(&raw).unwrap();
        summary_retina.sample_top_down(&raw).unwrap();
        assert_eq!(display_retina.readings(), summary_retina.readings());
        assert_eq!(display_retina.summary(), summary_retina.summary());
    }

    #[test]
    fn precomputed_samples_preserve_every_ommatidium_exactly() {
        let mut retina = FlyGymRetina::load(assets_dir()).unwrap();
        let sources = build_fisheye_source_pixels();
        for pattern in [0, 1, 17, 255] {
            let raw: Vec<u8> = (0..RGB_BYTES).map(|i| ((i * pattern + i / 7) % 256) as u8).collect();
            let mut expected = vec![[0.0; 2]; OMMATIDIA_PER_EYE];
            for (pixel, &id) in retina.ommatidia_id_map.iter().enumerate() {
                if id == 0 { continue; }
                let index = usize::from(id - 1);
                let channel = usize::from(retina.pale_mask[index]);
                let value = sources[pixel].map_or(0, |source| raw[source * 3 + channel + 1]);
                expected[index][channel] += f64::from(value) / f64::from(retina.pixels_per_ommatidium[index]);
            }
            for reading in &mut expected {
                reading[0] /= 255.0;
                reading[1] /= 255.0;
            }
            retina.sample_top_down(&raw).unwrap();
            assert_eq!(retina.readings(), expected);
        }
    }
}
