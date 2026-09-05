use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::npy;

const ARRAY_FILES: [&str; 4] = [
    "neuron_ids.npy",
    "row_ptr.npy",
    "destinations.npy",
    "signed_counts.npy",
];
const UINT32_MAX: u64 = u32::MAX as u64;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ManifestCounts {
    pub neurons: u64,
    pub edges: u64,
    pub contacts: u64,
    pub excitatory_edges: u64,
    pub inhibitory_edges: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PackManifest {
    pub schema_version: u64,
    pub materialization: String,
    pub neuron_count: u64,
    pub edge_count: u64,
    pub contact_sum: u64,
    pub excitatory_edge_count: u64,
    pub inhibitory_edge_count: u64,
    #[serde(default)]
    pub counts: Option<ManifestCounts>,
    #[serde(default)]
    pub source_sha256: BTreeMap<String, String>,
    #[serde(default)]
    pub array_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArrayCounts {
    neuron_count: u64,
    edge_count: u64,
    contact_sum: u64,
    excitatory_edge_count: u64,
    inhibitory_edge_count: u64,
}

#[derive(Debug, Clone)]
pub struct ConnectomePack {
    pub neuron_ids: Vec<u64>,
    pub row_ptr: Vec<u32>,
    pub destinations: Vec<u32>,
    pub signed_counts: Vec<i16>,
    pub manifest: PackManifest,
    pub path: Option<PathBuf>,
}

impl ConnectomePack {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref();
        if !root.is_dir() {
            bail!("connectome pack is not a directory: {}", root.display())
        }

        let manifest_path = root.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: PackManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("parsing {}", manifest_path.display()))?;
        if manifest.schema_version != 1 {
            bail!(
                "unsupported connectome schema version {}; expected 1",
                manifest.schema_version
            )
        }
        verify_array_hashes(root, &manifest)?;

        let neuron_ids = npy::read_u64(root.join("neuron_ids.npy"))?;
        let row_ptr = npy::read_u32(root.join("row_ptr.npy"))?;
        let destinations = npy::read_u32(root.join("destinations.npy"))?;
        let signed_counts = npy::read_i16(root.join("signed_counts.npy"))?;
        validate_arrays(
            &neuron_ids,
            &row_ptr,
            &destinations,
            &signed_counts,
            Some(&manifest),
        )?;

        Ok(Self {
            neuron_ids,
            row_ptr,
            destinations,
            signed_counts,
            manifest,
            path: Some(root.to_path_buf()),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path)
    }

    pub fn from_arrays<N, R, D, S>(
        neuron_ids: N,
        row_ptr: R,
        destinations: D,
        signed_counts: S,
    ) -> Result<Self>
    where
        N: AsRef<[u64]>,
        R: AsRef<[u32]>,
        D: AsRef<[u32]>,
        S: AsRef<[i16]>,
    {
        let neuron_ids = neuron_ids.as_ref().to_vec();
        let row_ptr = row_ptr.as_ref().to_vec();
        let destinations = destinations.as_ref().to_vec();
        let signed_counts = signed_counts.as_ref().to_vec();
        let counts = validate_arrays(&neuron_ids, &row_ptr, &destinations, &signed_counts, None)?;
        let manifest = PackManifest {
            schema_version: 1,
            materialization: "unknown".to_string(),
            neuron_count: counts.neuron_count,
            edge_count: counts.edge_count,
            contact_sum: counts.contact_sum,
            excitatory_edge_count: counts.excitatory_edge_count,
            inhibitory_edge_count: counts.inhibitory_edge_count,
            counts: Some(ManifestCounts {
                neurons: counts.neuron_count,
                edges: counts.edge_count,
                contacts: counts.contact_sum,
                excitatory_edges: counts.excitatory_edge_count,
                inhibitory_edges: counts.inhibitory_edge_count,
            }),
            source_sha256: BTreeMap::new(),
            array_sha256: BTreeMap::new(),
        };
        Ok(Self {
            neuron_ids,
            row_ptr,
            destinations,
            signed_counts,
            manifest,
            path: None,
        })
    }

    pub fn materialization(&self) -> &str {
        &self.manifest.materialization
    }

    pub fn neuron_count(&self) -> usize {
        self.neuron_ids.len()
    }

    pub fn edge_count(&self) -> usize {
        self.destinations.len()
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn verify_array_hashes(root: &Path, manifest: &PackManifest) -> Result<()> {
    if manifest.array_sha256.len() != ARRAY_FILES.len() {
        bail!("manifest array_sha256 must contain exactly four arrays")
    }
    for name in ARRAY_FILES {
        let expected = manifest
            .array_sha256
            .get(name)
            .with_context(|| format!("manifest is missing array hash for {name}"))?;
        if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("manifest has an invalid SHA256 for {name}")
        }
        let actual = sha256_file(&root.join(name))?;
        if actual != *expected {
            bail!("SHA256 mismatch for {name}")
        }
    }
    Ok(())
}

fn validate_arrays(
    neuron_ids: &[u64],
    row_ptr: &[u32],
    destinations: &[u32],
    signed_counts: &[i16],
    manifest: Option<&PackManifest>,
) -> Result<ArrayCounts> {
    let neuron_count = u64::try_from(neuron_ids.len()).context("neuron count overflow")?;
    let edge_count = u64::try_from(destinations.len()).context("edge count overflow")?;
    if neuron_count > UINT32_MAX {
        bail!("neuron count exceeds uint32 index limit")
    }
    if edge_count > UINT32_MAX {
        bail!("edge count exceeds uint32 row-pointer limit")
    }
    let expected_row_ptr_len = neuron_ids
        .len()
        .checked_add(1)
        .context("row pointer length overflow")?;
    if row_ptr.len() != expected_row_ptr_len {
        bail!(
            "row_ptr has length {}, expected {}",
            row_ptr.len(),
            expected_row_ptr_len
        )
    }
    if destinations.len() != signed_counts.len() {
        bail!(
            "destinations and signed_counts have different lengths: {} and {}",
            destinations.len(),
            signed_counts.len()
        )
    }
    if row_ptr.first().copied() != Some(0) {
        bail!("row_ptr must start at zero")
    }
    if row_ptr.last().copied() != Some(edge_count as u32) {
        bail!("row_ptr must end at the edge count")
    }
    if row_ptr.windows(2).any(|window| window[1] < window[0]) {
        bail!("row_ptr must be non-decreasing")
    }

    let neuron_count_u32 = neuron_count as u32;
    if destinations
        .iter()
        .any(|&destination| destination >= neuron_count_u32)
    {
        bail!("destination index is outside the neuron table")
    }
    if signed_counts.contains(&0) {
        bail!("zero-weight edges are not allowed")
    }

    let mut unique_ids = HashSet::with_capacity(neuron_ids.len());
    for &id in neuron_ids {
        if !unique_ids.insert(id) {
            bail!("neuron IDs must be unique")
        }
    }

    let contact_sum = signed_counts
        .iter()
        .map(|&count| i64::from(count).unsigned_abs())
        .try_fold(0u64, |sum, value| sum.checked_add(value))
        .context("contact sum overflow")?;
    let excitatory_edge_count =
        u64::try_from(signed_counts.iter().filter(|&&count| count > 0).count())?;
    let inhibitory_edge_count =
        u64::try_from(signed_counts.iter().filter(|&&count| count < 0).count())?;
    let counts = ArrayCounts {
        neuron_count,
        edge_count,
        contact_sum,
        excitatory_edge_count,
        inhibitory_edge_count,
    };

    if let Some(manifest) = manifest {
        if manifest.neuron_count != counts.neuron_count {
            bail!("manifest neuron_count does not match neuron_ids")
        }
        if manifest.edge_count != counts.edge_count {
            bail!("manifest edge_count does not match edge arrays")
        }
        if manifest.contact_sum != counts.contact_sum {
            bail!("manifest contact_sum does not match signed_counts")
        }
        if manifest.excitatory_edge_count != counts.excitatory_edge_count {
            bail!("manifest excitatory_edge_count does not match signed_counts")
        }
        if manifest.inhibitory_edge_count != counts.inhibitory_edge_count {
            bail!("manifest inhibitory_edge_count does not match signed_counts")
        }
        if let Some(manifest_counts) = &manifest.counts {
            let expected = ManifestCounts {
                neurons: counts.neuron_count,
                edges: counts.edge_count,
                contacts: counts.contact_sum,
                excitatory_edges: counts.excitatory_edge_count,
                inhibitory_edges: counts.inhibitory_edge_count,
            };
            if manifest_counts != &expected {
                bail!("manifest counts do not match packed arrays")
            }
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    const MAGIC: &[u8; 6] = b"\x93NUMPY";
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "flybrain-pack-{}-{nonce}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("create temporary directory");
        path
    }

    fn npy(descr: &str, shape: usize, data: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape},), }}")
                .into_bytes();
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut output = Vec::with_capacity(10 + header.len() + data.len());
        output.extend_from_slice(MAGIC);
        output.extend_from_slice(&[1, 0]);
        output.extend_from_slice(&(header.len() as u16).to_le_bytes());
        output.extend_from_slice(&header);
        output.extend_from_slice(data);
        output
    }

    fn write_u64(path: &Path, values: &[u64]) {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        fs::write(path, npy("<u8", values.len(), &data)).expect("write u64 NPY");
    }

    fn write_u32(path: &Path, values: &[u32]) {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        fs::write(path, npy("<u4", values.len(), &data)).expect("write u32 NPY");
    }

    fn write_i16(path: &Path, values: &[i16]) {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        fs::write(path, npy("<i2", values.len(), &data)).expect("write i16 NPY");
    }

    fn write_test_pack(root: &Path) {
        fs::create_dir(root).expect("create pack directory");
        write_u64(&root.join("neuron_ids.npy"), &[10, 20, 30]);
        write_u32(&root.join("row_ptr.npy"), &[0, 2, 3, 3]);
        write_u32(&root.join("destinations.npy"), &[2, 1, 0]);
        write_i16(&root.join("signed_counts.npy"), &[3, 1, -2]);

        let mut hashes = BTreeMap::new();
        for name in ARRAY_FILES {
            hashes.insert(name, sha256_file(&root.join(name)).expect("hash array"));
        }
        let manifest = serde_json::json!({
            "schema_version": 1,
            "materialization": "test",
            "neuron_count": 3,
            "edge_count": 3,
            "contact_sum": 6,
            "excitatory_edge_count": 2,
            "inhibitory_edge_count": 1,
            "counts": {
                "neurons": 3,
                "edges": 3,
                "contacts": 6,
                "excitatory_edges": 2,
                "inhibitory_edges": 1,
            },
            "source_sha256": {},
            "array_sha256": hashes,
        });
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    fn remove_directory(path: &Path) {
        fs::remove_dir_all(path).expect("remove temporary pack");
    }

    #[test]
    fn opens_and_validates_owned_pack() {
        let parent = temporary_directory();
        let root = parent.join("pack");
        write_test_pack(&root);

        let pack = ConnectomePack::open(&root).expect("open pack");
        assert_eq!(pack.materialization(), "test");
        assert_eq!(pack.neuron_ids, vec![10, 20, 30]);
        assert_eq!(pack.row_ptr, vec![0, 2, 3, 3]);
        assert_eq!(pack.destinations, vec![2, 1, 0]);
        assert_eq!(pack.signed_counts, vec![3, 1, -2]);
        assert_eq!(pack.neuron_count(), 3);
        assert_eq!(pack.edge_count(), 3);

        remove_directory(&parent);
    }

    #[test]
    fn from_arrays_builds_a_fixture_manifest() {
        let pack = ConnectomePack::from_arrays([10, 20], [0, 1, 1], [1], [-3]).expect("build pack");
        assert_eq!(pack.materialization(), "unknown");
        assert_eq!(pack.manifest.contact_sum, 3);
        assert_eq!(pack.manifest.inhibitory_edge_count, 1);
        assert!(pack.path.is_none());
    }

    #[test]
    fn rejects_tampered_array_before_loading_it() {
        let parent = temporary_directory();
        let root = parent.join("pack");
        write_test_pack(&root);
        let mut bytes = fs::read(root.join("signed_counts.npy")).expect("read array");
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(root.join("signed_counts.npy"), bytes).expect("tamper array");

        let error = ConnectomePack::open(&root).expect_err("tampered array must fail");
        assert!(error.to_string().contains("SHA256 mismatch"));
        remove_directory(&parent);
    }

    #[test]
    fn from_arrays_rejects_bad_csr() {
        let error = ConnectomePack::from_arrays([10], [0, 1], [1], [1])
            .expect_err("out-of-range destination must fail");
        assert!(error.to_string().contains("destination"));

        let error = ConnectomePack::from_arrays([10], [1, 1], [0], [1])
            .expect_err("nonzero row pointer start must fail");
        assert!(error.to_string().contains("start at zero"));
    }
}
