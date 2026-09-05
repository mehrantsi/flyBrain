use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const INPUT_SIZE: usize = 104;
pub const HIDDEN_SIZE: usize = 256;
pub const OUTPUT_SIZE: usize = 12;
pub const LAYER_NORM_EPSILON: f32 = 1e-5;
pub const SOURCE_ARCHIVE_SHA256: &str =
    "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582";

const INPUT_ORDER: [(&str, usize); 9] = [
    ("accelerometer", 3),
    ("actuator_activation", 0),
    ("gyro", 3),
    ("joints_pos", 25),
    ("joints_vel", 25),
    ("ref_displacement", 18),
    ("ref_root_quat", 24),
    ("velocimeter", 3),
    ("world_zaxis", 3),
];

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PolicyInputBlock {
    pub name: String,
    pub size: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PolicyTensorManifest {
    pub name: String,
    pub source_variable_index: usize,
    pub source_variable_name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset_f32: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PolicySource {
    pub figshare_doi: String,
    pub data_license: String,
    pub archive_sha256: String,
    pub archive_relative_path: String,
    pub saved_model_relative_path: String,
    pub saved_model_pb_sha256: String,
    pub tensorflow: String,
    pub tensorflow_probability: String,
    pub legacy_typespec_alias: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PolicyManifest {
    pub schema: String,
    pub schema_version: u64,
    pub policy: String,
    pub input_size: usize,
    pub input_order: Vec<PolicyInputBlock>,
    pub hidden_size: usize,
    pub output_size: usize,
    pub layer_norm_epsilon: f32,
    pub activations: Vec<String>,
    pub output_head: String,
    pub scale_head: String,
    pub weights_file: String,
    pub weights_sha256: String,
    pub weights_f32_count: usize,
    pub tensors: Vec<PolicyTensorManifest>,
    pub source: PolicySource,
    pub fixture_file: String,
}

pub struct FlyBodyFlightPolicy {
    manifest: PolicyManifest,
    torso_linear_weight: Box<[f32]>,
    torso_linear_bias: Box<[f32]>,
    layer_norm_offset: Box<[f32]>,
    layer_norm_scale: Box<[f32]>,
    mlp_linear_0_weight: Box<[f32]>,
    mlp_linear_0_bias: Box<[f32]>,
    mlp_linear_1_weight: Box<[f32]>,
    mlp_linear_1_bias: Box<[f32]>,
    mean_head_weight: Box<[f32]>,
    mean_head_bias: Box<[f32]>,
}

impl FlyBodyFlightPolicy {
    pub fn load(assets_dir: impl AsRef<Path>) -> Result<Self> {
        let assets_dir = assets_dir.as_ref();
        let manifest_path = assets_dir.join("flybody_flight_policy_v1.json");
        let manifest_bytes = fs::read(&manifest_path).with_context(|| {
            format!(
                "reading FlyBody policy manifest {}",
                manifest_path.display()
            )
        })?;
        let manifest: PolicyManifest =
            serde_json::from_slice(&manifest_bytes).with_context(|| {
                format!(
                    "parsing FlyBody policy manifest {}",
                    manifest_path.display()
                )
            })?;
        validate_manifest(&manifest)?;

        let weights_path = assets_dir.join(&manifest.weights_file);
        let weights_bytes = fs::read(&weights_path).with_context(|| {
            format!("reading FlyBody policy weights {}", weights_path.display())
        })?;
        if weights_bytes.len() != manifest.weights_f32_count * 4 {
            bail!(
                "FlyBody policy weights contain {} bytes, expected {}",
                weights_bytes.len(),
                manifest.weights_f32_count * 4
            )
        }
        let actual_hash = sha256(&weights_bytes);
        if actual_hash != manifest.weights_sha256 {
            bail!(
                "FlyBody policy weights SHA256 {} does not match manifest {}",
                actual_hash,
                manifest.weights_sha256
            )
        }
        let weights = weights_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("validated f32 chunk")))
            .collect::<Vec<_>>();

        let tensors = manifest
            .tensors
            .iter()
            .map(|tensor| {
                let end = tensor
                    .offset_f32
                    .checked_add(tensor.count)
                    .context("FlyBody policy tensor offset overflow")?;
                if end > weights.len() {
                    bail!(
                        "FlyBody policy tensor {} exceeds the weights array",
                        tensor.name
                    )
                }
                Ok(weights[tensor.offset_f32..end].to_vec().into_boxed_slice())
            })
            .collect::<Result<Vec<_>>>()?;
        let by_name = tensors
            .into_iter()
            .zip(&manifest.tensors)
            .map(|(values, tensor)| (tensor.name.clone(), values))
            .collect::<std::collections::HashMap<_, _>>();

        Ok(Self {
            manifest,
            torso_linear_weight: take_tensor(&by_name, "torso_linear_weight")?,
            torso_linear_bias: take_tensor(&by_name, "torso_linear_bias")?,
            layer_norm_offset: take_tensor(&by_name, "layer_norm_offset")?,
            layer_norm_scale: take_tensor(&by_name, "layer_norm_scale")?,
            mlp_linear_0_weight: take_tensor(&by_name, "mlp_linear_0_weight")?,
            mlp_linear_0_bias: take_tensor(&by_name, "mlp_linear_0_bias")?,
            mlp_linear_1_weight: take_tensor(&by_name, "mlp_linear_1_weight")?,
            mlp_linear_1_bias: take_tensor(&by_name, "mlp_linear_1_bias")?,
            mean_head_weight: take_tensor(&by_name, "mean_head_weight")?,
            mean_head_bias: take_tensor(&by_name, "mean_head_bias")?,
        })
    }

    pub fn manifest(&self) -> &PolicyManifest {
        &self.manifest
    }

    pub fn infer(&self, input: &[f32]) -> Result<[f32; OUTPUT_SIZE]> {
        if input.len() != INPUT_SIZE {
            bail!(
                "FlyBody flight policy input has {} values, expected {}",
                input.len(),
                INPUT_SIZE
            )
        }

        let mut torso = [0.0_f32; HIDDEN_SIZE];
        for (output, torso_value) in torso.iter_mut().enumerate() {
            let mut value = self.torso_linear_bias[output];
            for (feature, &input_value) in input.iter().enumerate() {
                value += input_value * self.torso_linear_weight[feature * HIDDEN_SIZE + output];
            }
            *torso_value = value;
        }

        let mean = torso.iter().copied().sum::<f32>() / HIDDEN_SIZE as f32;
        let variance = torso
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / HIDDEN_SIZE as f32;
        let inverse_standard_deviation = (variance + LAYER_NORM_EPSILON).sqrt().recip();
        for (index, torso_value) in torso.iter_mut().enumerate() {
            *torso_value =
                ((*torso_value - mean) * inverse_standard_deviation * self.layer_norm_scale[index]
                    + self.layer_norm_offset[index])
                    .tanh();
        }

        apply_linear_elu(
            &mut torso,
            &self.mlp_linear_0_weight,
            &self.mlp_linear_0_bias,
        );
        apply_linear_elu(
            &mut torso,
            &self.mlp_linear_1_weight,
            &self.mlp_linear_1_bias,
        );

        let mut output = [0.0_f32; OUTPUT_SIZE];
        for (action, output_value) in output.iter_mut().enumerate() {
            let mut value = self.mean_head_bias[action];
            for (hidden, &torso_value) in torso.iter().enumerate() {
                value += torso_value * self.mean_head_weight[hidden * OUTPUT_SIZE + action];
            }
            *output_value = value;
        }
        Ok(output)
    }
}

fn apply_linear_elu(state: &mut [f32; HIDDEN_SIZE], weight: &[f32], bias: &[f32]) {
    let input = *state;
    for output in 0..HIDDEN_SIZE {
        let mut value = bias[output];
        for hidden in 0..HIDDEN_SIZE {
            value += input[hidden] * weight[hidden * HIDDEN_SIZE + output];
        }
        state[output] = if value >= 0.0 {
            value
        } else {
            value.exp() - 1.0
        };
    }
}

fn take_tensor(
    tensors: &std::collections::HashMap<String, Box<[f32]>>,
    name: &str,
) -> Result<Box<[f32]>> {
    tensors
        .get(name)
        .cloned()
        .with_context(|| format!("FlyBody policy manifest is missing tensor {name}"))
}

fn validate_manifest(manifest: &PolicyManifest) -> Result<()> {
    if manifest.schema != "flybody.flight-policy" || manifest.schema_version != 1 {
        bail!("unsupported FlyBody policy manifest schema")
    }
    if manifest.policy != "flight"
        || manifest.input_size != INPUT_SIZE
        || manifest.hidden_size != HIDDEN_SIZE
        || manifest.output_size != OUTPUT_SIZE
        || manifest.layer_norm_epsilon != LAYER_NORM_EPSILON
        || manifest.activations != ["tanh", "elu", "elu"]
        || manifest.output_head != "mean"
        || manifest.scale_head != "omitted"
    {
        bail!("FlyBody policy manifest does not describe the pinned graph")
    }
    let expected_input_order = INPUT_ORDER
        .into_iter()
        .map(|(name, size)| PolicyInputBlock {
            name: name.to_string(),
            size,
        })
        .collect::<Vec<_>>();
    if manifest.input_order != expected_input_order {
        bail!("FlyBody policy input order does not match the pinned 104-feature layout")
    }
    if manifest.source.figshare_doi != "10.25378/janelia.25309105.v4"
        || manifest.source.data_license != "GPL-3.0+"
        || manifest.source.archive_sha256 != SOURCE_ARCHIVE_SHA256
        || manifest.source.tensorflow != "2.18.1"
        || manifest.source.tensorflow_probability != "0.25.0"
        || manifest.source.legacy_typespec_alias
            != "tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"
    {
        bail!("FlyBody policy provenance is not pinned to the requested source")
    }
    validate_sha256(&manifest.weights_sha256, "weights_sha256")?;
    validate_sha256(
        &manifest.source.saved_model_pb_sha256,
        "saved_model_pb_sha256",
    )?;
    if manifest.weights_f32_count == 0 || manifest.tensors.is_empty() {
        bail!("FlyBody policy manifest has no weight tensors")
    }

    let expected_tensors = [
        ("torso_linear_weight", vec![104, 256], 26624),
        ("torso_linear_bias", vec![256], 256),
        ("layer_norm_offset", vec![256], 256),
        ("layer_norm_scale", vec![256], 256),
        ("mlp_linear_0_weight", vec![256, 256], 65536),
        ("mlp_linear_0_bias", vec![256], 256),
        ("mlp_linear_1_weight", vec![256, 256], 65536),
        ("mlp_linear_1_bias", vec![256], 256),
        ("mean_head_weight", vec![256, 12], 3072),
        ("mean_head_bias", vec![12], 12),
    ];
    if manifest.tensors.len() != expected_tensors.len() {
        bail!("FlyBody policy manifest has an unexpected tensor count")
    }
    let mut names = HashSet::with_capacity(manifest.tensors.len());
    let mut expected_offset = 0;
    for (tensor, (name, shape, count)) in manifest.tensors.iter().zip(expected_tensors) {
        if tensor.name != name
            || tensor.dtype != "f32le"
            || tensor.shape != shape
            || tensor.count != count
            || tensor.offset_f32 != expected_offset
            || !names.insert(tensor.name.as_str())
        {
            bail!("FlyBody policy tensor manifest is not the pinned layout")
        }
        expected_offset += count;
    }
    if expected_offset != manifest.weights_f32_count {
        bail!("FlyBody policy tensor counts do not match the weights array")
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("FlyBody policy {name} is not a SHA256 digest")
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct Fixture {
        schema: String,
        schema_version: u64,
        inputs_flat_f32: Vec<Vec<f32>>,
        expected_mean_f32: Vec<Vec<f32>>,
        tolerance_abs: f32,
    }

    fn assets_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/neuromechfly")
    }

    #[test]
    fn loads_pinned_manifest_and_weights() {
        let policy = FlyBodyFlightPolicy::load(assets_dir()).unwrap();
        assert_eq!(policy.manifest().input_size, INPUT_SIZE);
        assert_eq!(policy.manifest().weights_f32_count, 162_060);
        assert_eq!(
            policy.manifest().source.archive_sha256,
            SOURCE_ARCHIVE_SHA256
        );
    }

    #[test]
    fn matches_tensorflow_fixture() {
        let assets = assets_dir();
        let policy = FlyBodyFlightPolicy::load(&assets).unwrap();
        let fixture: Fixture = serde_json::from_slice(
            &fs::read(assets.join("flybody_flight_policy_fixture_v1.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(fixture.schema, "flybody.flight-policy-fixture");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.inputs_flat_f32.len(),
            fixture.expected_mean_f32.len()
        );
        let mut maximum_error = 0.0_f32;
        for (input, expected) in fixture
            .inputs_flat_f32
            .iter()
            .zip(&fixture.expected_mean_f32)
        {
            let actual = policy.infer(input).unwrap();
            assert_eq!(expected.len(), OUTPUT_SIZE);
            for (actual, expected) in actual.into_iter().zip(expected) {
                maximum_error = maximum_error.max((actual - *expected).abs());
            }
        }
        assert!(
            maximum_error <= fixture.tolerance_abs,
            "maximum error {maximum_error}"
        );
    }

    #[test]
    fn rejects_wrong_input_shape() {
        let policy = FlyBodyFlightPolicy::load(assets_dir()).unwrap();
        let error = policy
            .infer(&[0.0; INPUT_SIZE - 1])
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected 104"));
    }
}
