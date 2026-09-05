use std::fs;
use std::path::Path;
use std::str;

use anyhow::{Context, Result, bail};

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_HEADER_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct ParsedNpy {
    bytes: Vec<u8>,
    data_offset: usize,
    descr: String,
}

#[derive(Debug)]
struct Header {
    descr: String,
    length: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn skip_space(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.bytes.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: u8) -> Result<()> {
        if self.consume(expected) {
            Ok(())
        } else {
            let found = self
                .bytes
                .get(self.position)
                .copied()
                .map_or("end of header".to_string(), |byte| format!("byte {byte:?}"));
            bail!("expected {:?}, found {found}", expected as char)
        }
    }

    fn quoted(&mut self) -> Result<String> {
        self.expect(b'\'')?;
        let start = self.position;
        while let Some(byte) = self.bytes.get(self.position) {
            if *byte == b'\'' {
                let value = str::from_utf8(&self.bytes[start..self.position])
                    .context("NPY header contains a non-UTF-8 string")?
                    .to_owned();
                self.position += 1;
                return Ok(value);
            }
            if *byte == b'\\' {
                bail!("escaped characters are not supported in an NPY header string");
            }
            self.position += 1;
        }
        bail!("unterminated string in NPY header")
    }

    fn identifier(&mut self) -> Result<String> {
        let start = self.position;
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            self.position += 1;
        }
        if self.position == start {
            bail!("expected an NPY header identifier")
        }
        Ok(str::from_utf8(&self.bytes[start..self.position])?.to_owned())
    }

    fn unsigned_integer(&mut self) -> Result<u64> {
        let start = self.position;
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            self.position += 1;
        }
        if self.position == start {
            bail!("expected a non-negative integer in NPY header")
        }
        let value = str::from_utf8(&self.bytes[start..self.position])?.parse::<u64>()?;
        Ok(value)
    }

    fn shape(&mut self) -> Result<usize> {
        self.expect(b'(')?;
        self.skip_space();
        let value = self.unsigned_integer()?;
        self.skip_space();
        if !self.consume(b',') {
            bail!("NPY array must have a one-dimensional shape")
        }
        self.skip_space();
        if !self.consume(b')') {
            bail!("NPY array must have a one-dimensional shape")
        }
        usize::try_from(value).context("NPY shape does not fit in usize")
    }

    fn header(mut self) -> Result<Header> {
        self.skip_space();
        self.expect(b'{')?;

        let mut descr = None;
        let mut fortran_order = None;
        let mut length = None;

        loop {
            self.skip_space();
            if self.consume(b'}') {
                break;
            }
            let key = self.quoted()?;
            self.skip_space();
            self.expect(b':')?;
            self.skip_space();
            match key.as_str() {
                "descr" => {
                    if descr.is_some() {
                        bail!("duplicate descr field in NPY header")
                    }
                    descr = Some(self.quoted()?);
                }
                "fortran_order" => {
                    if fortran_order.is_some() {
                        bail!("duplicate fortran_order field in NPY header")
                    }
                    let value = self.identifier()?;
                    if value != "False" {
                        bail!("NPY array must use C order")
                    }
                    fortran_order = Some(false);
                }
                "shape" => {
                    if length.is_some() {
                        bail!("duplicate shape field in NPY header")
                    }
                    length = Some(self.shape()?);
                }
                _ => bail!("unsupported field {key:?} in NPY header"),
            }
            self.skip_space();
            if !self.consume(b',') {
                self.skip_space();
                self.expect(b'}')?;
                break;
            }
        }

        self.skip_space();
        if self.position != self.bytes.len() {
            bail!("unexpected data after NPY header dictionary")
        }
        if fortran_order != Some(false) {
            bail!("NPY header is missing fortran_order=False")
        }
        Ok(Header {
            descr: descr.context("NPY header is missing descr")?,
            length: length.context("NPY header is missing shape")?,
        })
    }
}

fn parse(path: &Path, expected_descr: &str, item_size: usize) -> Result<ParsedNpy> {
    let bytes = fs::read(path).with_context(|| format!("reading NPY file {}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        bail!("{} is not an NPY file", path.display())
    }
    if bytes[6] != 1 || bytes[7] != 0 {
        bail!(
            "{} uses unsupported NPY version {}.{}; expected 1.0",
            path.display(),
            bytes[6],
            bytes[7]
        )
    }

    let header_length = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    if header_length > MAX_HEADER_BYTES {
        bail!("{} has an oversized NPY header", path.display())
    }
    let header_start = 10usize;
    let data_offset = header_start
        .checked_add(header_length)
        .context("NPY header offset overflow")?;
    if data_offset > bytes.len() {
        bail!("{} is truncated before its NPY data", path.display())
    }
    if data_offset % 16 != 0 {
        bail!("{} has an unaligned NPY data offset", path.display())
    }

    let header_bytes = &bytes[header_start..data_offset];
    let header = Cursor::new(header_bytes)
        .header()
        .with_context(|| format!("parsing NPY header in {}", path.display()))?;
    if header.descr != expected_descr {
        bail!(
            "{} has dtype {}, expected {}",
            path.display(),
            header.descr,
            expected_descr
        )
    }
    let data_length = header
        .length
        .checked_mul(item_size)
        .context("NPY data length overflow")?;
    let expected_file_length = data_offset
        .checked_add(data_length)
        .context("NPY file length overflow")?;
    if bytes.len() != expected_file_length {
        bail!(
            "{} has {} data bytes, expected {}",
            path.display(),
            bytes.len().saturating_sub(data_offset),
            data_length
        )
    }

    Ok(ParsedNpy {
        bytes,
        data_offset,
        descr: header.descr,
    })
}

fn read_values<T, F>(
    path: impl AsRef<Path>,
    expected_descr: &str,
    item_size: usize,
    decode: F,
) -> Result<Vec<T>>
where
    F: Fn(&[u8]) -> T,
{
    let path = path.as_ref();
    let parsed = parse(path, expected_descr, item_size)?;
    debug_assert_eq!(parsed.descr, expected_descr);
    let data = &parsed.bytes[parsed.data_offset..];
    Ok(data.chunks_exact(item_size).map(decode).collect())
}

pub fn read_u64(path: impl AsRef<Path>) -> Result<Vec<u64>> {
    read_values(path, "<u8", 8, |bytes| {
        u64::from_le_bytes(bytes.try_into().expect("validated u64 chunk"))
    })
}

pub fn read_u32(path: impl AsRef<Path>) -> Result<Vec<u32>> {
    read_values(path, "<u4", 4, |bytes| {
        u32::from_le_bytes(bytes.try_into().expect("validated u32 chunk"))
    })
}

pub fn read_i16(path: impl AsRef<Path>) -> Result<Vec<i16>> {
    read_values(path, "<i2", 2, |bytes| {
        i16::from_le_bytes(bytes.try_into().expect("validated i16 chunk"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_file(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "flybrain-npy-{}-{nonce}-{name}",
            std::process::id()
        ))
    }

    fn npy_with_shape(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}")
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

    fn npy(descr: &str, shape: usize, data: &[u8]) -> Vec<u8> {
        npy_with_shape(descr, &format!("({shape},)"), data)
    }

    #[test]
    fn reads_compiler_contract_u32() {
        let path = temporary_file("u32.npy");
        let data = [0u32, 1, 0xfeed_beef];
        let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
        fs::write(&path, npy("<u4", data.len(), &bytes)).expect("write NPY");

        assert_eq!(read_u32(&path).expect("read NPY"), data);
        fs::remove_file(path).expect("remove NPY");
    }

    #[test]
    fn rejects_wrong_dtype_and_trailing_bytes() {
        let dtype_path = temporary_file("dtype.npy");
        fs::write(&dtype_path, npy("<i4", 1, &[0, 0, 0, 0])).expect("write NPY");
        let error = read_u32(&dtype_path).expect_err("wrong dtype must fail");
        assert!(error.to_string().contains("dtype"));

        let trailing_path = temporary_file("trailing.npy");
        let mut bytes = npy("<u4", 1, &[0, 0, 0, 0]);
        bytes.push(1);
        fs::write(&trailing_path, bytes).expect("write NPY");
        let error = read_u32(&trailing_path).expect_err("trailing bytes must fail");
        assert!(error.to_string().contains("data bytes"));

        fs::remove_file(dtype_path).expect("remove NPY");
        fs::remove_file(trailing_path).expect("remove NPY");
    }

    #[test]
    fn rejects_fortran_order_and_non_one_dimensional_shape() {
        let fortran_path = temporary_file("fortran.npy");
        let mut bytes = npy("<u8", 1, &[0; 8]);
        let header_start = 10;
        let header_end = header_start + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        let header =
            String::from_utf8(bytes[header_start..header_end].to_vec()).expect("ASCII header");
        let header = header.replace("False", "True ");
        bytes[header_start..header_end].copy_from_slice(header.as_bytes());
        fs::write(&fortran_path, bytes).expect("write NPY");
        let error = read_u64(&fortran_path).expect_err("Fortran order must fail");
        assert!(format!("{error:#}").contains("C order"));

        let shape_path = temporary_file("shape.npy");
        let bytes = npy_with_shape("<u8", "(1, 1)", &[0; 8]);
        fs::write(&shape_path, bytes).expect("write NPY");
        let error = read_u64(&shape_path).expect_err("two-dimensional shape must fail");
        assert!(format!("{error:#}").contains("one-dimensional"));

        fs::remove_file(fortran_path).expect("remove NPY");
        fs::remove_file(shape_path).expect("remove NPY");
    }
}
