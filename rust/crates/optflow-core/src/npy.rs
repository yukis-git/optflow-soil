//! NPY array reading and writing.
//!
//! Works on byte slices only (no OS I/O). Reads little-endian arrays of dtype `<f8`, `<f4`, `|b1`,
//! `|u1` and `<u2` from format versions 1.0, 2.0 and 3.0, in C order or Fortran order (a Fortran
//! order file is reordered to C order, as `numpy.load` shows the same array); everything else
//! (big-endian, other dtypes) is an error. Writes C-order `<f4` arrays as version 1.0 with the
//! header padded to a multiple of 64 bytes, byte-for-byte what `numpy.save` produces for any shape
//! numpy can build (the header of such an array always fits version 1.0).

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Contents of an `.npy` file, flattened in C order.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

/// Element storage, one variant per supported dtype.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    U8(Vec<u8>),
    U16(Vec<u16>),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NpyError {
    #[error("not an npy file (bad magic)")]
    BadMagic,
    #[error("unsupported npy format version {0}.{1}")]
    UnsupportedVersion(u8, u8),
    #[error("npy data is truncated")]
    Truncated,
    #[error("malformed npy header: {0}")]
    BadHeader(String),
    #[error("unsupported npy dtype {0:?} (supported: <f8, <f4, |b1, |u1, <u2)")]
    UnsupportedDtype(String),
    #[error("npy payload is {actual} bytes but the header implies {expected}")]
    SizeMismatch { expected: usize, actual: usize },
}

#[derive(Clone, Copy)]
enum Dtype {
    F64,
    F32,
    Bool,
    U8,
    U16,
}

impl Dtype {
    fn parse(descr: &str) -> Result<Self, NpyError> {
        match descr {
            "<f8" => Ok(Self::F64),
            "<f4" => Ok(Self::F32),
            "|b1" => Ok(Self::Bool),
            "|u1" => Ok(Self::U8),
            "<u2" => Ok(Self::U16),
            _ => Err(NpyError::UnsupportedDtype(descr.to_string())),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::Bool | Self::U8 => 1,
            Self::U16 => 2,
        }
    }
}

/// Total header length (magic, version, length field, dict text, space padding, newline) for a
/// version 1.0 file whose dict text is `dict_len` bytes: the next multiple of 64 above the 10-byte
/// prefix, the dict and the closing newline. As in numpy, which pads with 1 to 64 spaces, a header
/// that would end exactly on a multiple of 64 gets 64 more bytes.
pub fn header_len(dict_len: usize) -> usize {
    (10 + dict_len + 1) / 64 * 64 + 64
}

/// Spaces numpy keeps in a header for the first axis to grow, `GROWTH_AXIS_MAX_DIGITS` (the
/// digits of the largest size) minus the digits of the first size. They count when the header is
/// padded to a multiple of 64, so a shape of 15 or more dimensions gets a longer header than its
/// dict alone needs.
const GROWTH_AXIS_DIGITS: usize = 21;

/// Serialises a C-order `f32` array as an `<f4` version 1.0 `.npy` file: [`f32_header`], then
/// the values in little-endian order.
///
/// # Panics
/// If `data.len()` differs from the product of `shape`.
pub fn write_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
    let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    assert_eq!(
        count,
        Some(data.len()),
        "npy: shape {shape:?} does not match data length"
    );
    let mut out = f32_header(shape);
    out.reserve_exact(4 * data.len());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The header of [`write_f32`] for an array of `shape`: what precedes the values, so that a
/// large array can be written in pieces.
pub fn f32_header(shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            format!("({})", dims.join(", "))
        }
    };
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    let spare = shape.first().map_or(0, |n| {
        GROWTH_AXIS_DIGITS.saturating_sub(n.to_string().len())
    });
    let total = header_len(dict.len() + spare);
    let field = u16::try_from(total - 10).expect("npy: header too long for format version 1.0");
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&field.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    out
}

/// Parses an `.npy` file held in memory.
pub fn read(bytes: &[u8]) -> Result<NpyArray, NpyError> {
    if !bytes.starts_with(MAGIC) {
        return Err(NpyError::BadMagic);
    }
    let version = bytes.get(6..8).ok_or(NpyError::Truncated)?;
    let (major, minor) = (version[0], version[1]);
    let (field_len, prefix): (usize, usize) = match major {
        1 => (2, 10),
        2 | 3 => (4, 12),
        _ => return Err(NpyError::UnsupportedVersion(major, minor)),
    };
    if minor != 0 {
        return Err(NpyError::UnsupportedVersion(major, minor));
    }
    let field = bytes.get(8..8 + field_len).ok_or(NpyError::Truncated)?;
    let header_bytes = field
        .iter()
        .rev()
        .fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
    let end = prefix
        .checked_add(header_bytes)
        .ok_or(NpyError::Truncated)?;
    let header = bytes.get(prefix..end).ok_or(NpyError::Truncated)?;
    let payload = &bytes[end..];
    let header = std::str::from_utf8(header)
        .map_err(|_| NpyError::BadHeader("header is not valid UTF-8".into()))?;
    let (descr, fortran_order, shape) = parse_dict(header)?;
    let dtype = Dtype::parse(&descr)?;
    let count = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| NpyError::BadHeader("shape overflows usize".into()))?;
    let expected = count
        .checked_mul(dtype.size())
        .ok_or_else(|| NpyError::BadHeader("shape overflows usize".into()))?;
    if payload.len() != expected {
        return Err(NpyError::SizeMismatch {
            expected,
            actual: payload.len(),
        });
    }
    let reordered;
    let payload = if fortran_order && shape.len() > 1 {
        reordered = fortran_to_c(payload, dtype.size(), &shape);
        &reordered[..]
    } else {
        payload
    };
    let data = match dtype {
        Dtype::F64 => NpyData::F64(decode(payload, f64::from_le_bytes)),
        Dtype::F32 => NpyData::F32(decode(payload, f32::from_le_bytes)),
        Dtype::Bool => NpyData::Bool(payload.iter().map(|&b| b != 0).collect()),
        Dtype::U8 => NpyData::U8(payload.to_vec()),
        Dtype::U16 => NpyData::U16(decode(payload, u16::from_le_bytes)),
    };
    Ok(NpyArray { shape, data })
}

/// Reorders the elements (`size` bytes each) of `payload`, which holds an array of `shape` with
/// the first index varying fastest (Fortran order), into C order.
fn fortran_to_c(payload: &[u8], size: usize, shape: &[usize]) -> Vec<u8> {
    // how far apart in the file two elements are whose index differs by one along each axis; the
    // products are the prefixes of the element count, which `read` has checked
    let mut step = vec![1usize; shape.len()];
    for d in 1..shape.len() {
        step[d] = step[d - 1] * shape[d - 1];
    }
    let mut index = vec![0usize; shape.len()];
    let mut out = Vec::with_capacity(payload.len());
    for _ in 0..payload.len() / size {
        let at: usize = index.iter().zip(&step).map(|(i, s)| i * s).sum();
        out.extend_from_slice(&payload[at * size..(at + 1) * size]);
        // the next index in C order: the last axis runs fastest
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

fn decode<const N: usize, T>(payload: &[u8], from_le: fn([u8; N]) -> T) -> Vec<T> {
    let (chunks, _) = payload.as_chunks::<N>();
    chunks.iter().map(|&chunk| from_le(chunk)).collect()
}

/// Parses the header's Python dict literal into `(descr, fortran_order, shape)`.
fn parse_dict(text: &str) -> Result<(String, bool, Vec<usize>), NpyError> {
    let bad = |msg: &str| NpyError::BadHeader(msg.to_string());
    let inner = text
        .trim()
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .ok_or_else(|| bad("header is not a dict literal"))?;
    let (mut descr, mut fortran, mut shape) = (None, None, None);
    let mut rest = inner.trim_start();
    while !rest.is_empty() {
        let (key, after) = take_quoted(rest)?;
        let after = after
            .trim_start()
            .strip_prefix(':')
            .ok_or_else(|| bad("expected ':' after key"))?
            .trim_start();
        let (value, after) = take_value(after)?;
        let duplicate = || NpyError::BadHeader(format!("duplicate key {key:?}"));
        match (key, value) {
            ("descr", Value::Str(s)) => {
                if descr.replace(s.to_string()).is_some() {
                    return Err(duplicate());
                }
            }
            ("fortran_order", Value::Bare(b)) => {
                let flag = match b {
                    "True" => true,
                    "False" => false,
                    _ => return Err(bad("fortran_order is not True/False")),
                };
                if fortran.replace(flag).is_some() {
                    return Err(duplicate());
                }
            }
            ("shape", Value::Tuple(t)) => {
                if shape.replace(parse_shape(t)?).is_some() {
                    return Err(duplicate());
                }
            }
            ("descr" | "fortran_order" | "shape", _) => {
                return Err(NpyError::BadHeader(format!(
                    "unexpected value type for key {key:?}"
                )));
            }
            _ => return Err(NpyError::BadHeader(format!("unknown key {key:?}"))),
        }
        rest = after.trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r.trim_start();
        } else if !rest.is_empty() {
            return Err(bad("expected ',' between entries"));
        }
    }
    match (descr, fortran, shape) {
        (Some(d), Some(f), Some(s)) => Ok((d, f, s)),
        _ => Err(bad("header must have descr, fortran_order and shape")),
    }
}

enum Value<'a> {
    Str(&'a str),
    Tuple(&'a str),
    Bare(&'a str),
}

/// Reads a `'quoted'` (or `"quoted"`) string, returning its contents and the remaining text.
fn take_quoted(s: &str) -> Result<(&str, &str), NpyError> {
    let quote = match s.chars().next() {
        Some(q @ ('\'' | '"')) => q,
        _ => return Err(NpyError::BadHeader("expected a quoted string".into())),
    };
    let body = &s[1..];
    let close = body
        .find(quote)
        .ok_or_else(|| NpyError::BadHeader("unterminated string".into()))?;
    Ok((&body[..close], &body[close + 1..]))
}

fn take_value(s: &str) -> Result<(Value<'_>, &str), NpyError> {
    match s.chars().next() {
        Some('\'' | '"') => take_quoted(s).map(|(v, r)| (Value::Str(v), r)),
        Some('(') => {
            let close = s
                .find(')')
                .ok_or_else(|| NpyError::BadHeader("unterminated tuple".into()))?;
            Ok((Value::Tuple(&s[1..close]), &s[close + 1..]))
        }
        Some('[') => Err(NpyError::UnsupportedDtype("structured dtype".into())),
        Some(_) => {
            let end = s.find(',').unwrap_or(s.len());
            Ok((Value::Bare(s[..end].trim()), &s[end..]))
        }
        None => Err(NpyError::BadHeader("missing value".into())),
    }
}

/// Parses the inside of a shape tuple: `""`, `"5,"`, `"2, 3"` or `"2, 3,"`.
fn parse_shape(inner: &str) -> Result<Vec<usize>, NpyError> {
    let mut parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.last() == Some(&"") {
        parts.pop();
    }
    parts
        .into_iter()
        .map(|p| {
            p.parse::<usize>()
                .map_err(|_| NpyError::BadHeader(format!("bad shape entry {p:?}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Version 1.0 file with `dict` padded the way numpy does it.
    fn file(dict: &str, payload: &[u8]) -> Vec<u8> {
        let total = header_len(dict.len());
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&u16::try_from(total - 10).unwrap().to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.resize(total - 1, b' ');
        out.push(b'\n');
        out.extend_from_slice(payload);
        out
    }

    fn dict(descr: &str, shape: &str) -> String {
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}")
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    /// `numpy.save(f, np.array([1.5, -2.5, np.inf], dtype='<f4'))` (numpy 2.5.4).
    const NUMPY_F4: &str = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000c03f000020c00000807f";

    #[test]
    fn npy_header_len_examples() {
        assert_eq!(header_len(0), 64);
        assert_eq!(header_len(52), 64);
        // 10 + 53 + 1 is 64: numpy still adds a block (shapes found by fuzzing against numpy)
        assert_eq!(header_len(53), 128);
        assert_eq!(header_len(54), 128);
        assert_eq!(header_len(116), 128);
        assert_eq!(header_len(117), 192);
        assert_eq!(header_len(118), 192);
    }

    #[test]
    fn npy_matches_numpy_written_bytes() {
        let bytes = hex(NUMPY_F4);
        assert_eq!(bytes.len() % 64, 12);
        let arr = read(&bytes).unwrap();
        assert_eq!(arr.shape, vec![3]);
        assert_eq!(arr.data, NpyData::F32(vec![1.5, -2.5, f32::INFINITY]));
        assert_eq!(write_f32(&[3], &[1.5, -2.5, f32::INFINITY]), bytes);
    }

    #[test]
    fn npy_write_header_is_aligned_and_numpy_formatted() {
        for shape in [vec![], vec![0], vec![7], vec![2, 3], vec![1, 2, 3, 4]] {
            let n: usize = shape.iter().product();
            let bytes = write_f32(&shape, &vec![0.25; n]);
            let hlen = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
            assert_eq!(hlen % 64, 0);
            assert_eq!(bytes.len(), hlen + 4 * n);
            assert_eq!(bytes[hlen - 1], b'\n');
            assert!(bytes.starts_with(b"\x93NUMPY\x01\x00"));
        }
        let bytes = write_f32(&[2, 3], &[0.0; 6]);
        let text = std::str::from_utf8(&bytes[10..]).unwrap();
        assert!(text.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"));
        assert!(write_f32(&[5], &[0.0; 5]).windows(5).any(|w| w == b"(5,),"));
        assert!(write_f32(&[], &[1.0]).windows(3).any(|w| w == b"(),"));
    }

    /// The header length of `numpy.save` for `<f4` arrays of these shapes (numpy 2.5.4). numpy
    /// reserves `21 - len(str(shape[0]))` spaces for the first axis to grow before it pads the
    /// header to a multiple of 64, so deep shapes get a longer header than the dict alone needs
    /// (from 15 dimensions of one-digit sizes on).
    const NUMPY_HEADER_LEN: [(&[usize], usize); 20] = [
        (&[], 128),
        (&[3], 128),
        (&[1234567, 0], 128),
        (&[2, 3], 128),
        (&[1, 2, 3, 4], 128),
        (&[1; 14], 128),
        (&[1; 15], 192),
        (&[1; 21], 192),
        (&[1; 22], 192),
        (&[1; 32], 192),
        (
            &[
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0,
            ],
            192,
        ),
        (&[1_000_000, 1_000_000, 1_000_000, 0], 128),
        (&[0; 15], 192),
        (&[123_456_789_012, 5, 5, 5, 5, 5, 0], 128),
        (
            &[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
            192,
        ),
        (&[1; 40], 256),
        (&[1; 64], 320),
        // a header that ends exactly on a multiple of 64 gets another block
        (&[3, 10, 2, 2, 5, 10, 2, 5, 6_908_023, 2, 1, 0], 192),
        (&[5, 9, 1, 1000, 5, 1, 1, 99, 99, 3, 3, 5, 0], 192),
        (&[1, 9, 5, 2, 5, 1, 1, 186_178_485_170, 1000, 0], 192),
    ];

    #[test]
    fn npy_write_header_length_matches_numpy_for_deep_shapes() {
        for (shape, want) in NUMPY_HEADER_LEN {
            let count: usize = shape.iter().product();
            let bytes = write_f32(shape, &vec![0.0; count]);
            let hlen = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
            assert_eq!(hlen, want, "shape {shape:?}");
            assert_eq!(bytes.len(), hlen + 4 * count);
        }
    }

    #[test]
    fn npy_roundtrip_f32() {
        let data: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.5,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-30,
            6.5,
            2.0,
            5.0,
        ];
        for shape in [vec![12], vec![3, 4], vec![2, 3, 2], vec![1, 12, 1]] {
            let arr = read(&write_f32(&shape, &data)).unwrap();
            assert_eq!(arr.shape, shape);
            assert_eq!(arr.data, NpyData::F32(data.clone()));
        }
    }

    #[test]
    fn npy_roundtrip_nan_and_empty() {
        let arr = read(&write_f32(&[2], &[f32::NAN, 1.0])).unwrap();
        match arr.data {
            NpyData::F32(v) => assert!(v[0].is_nan() && v[1] == 1.0),
            other => panic!("unexpected {other:?}"),
        }
        let arr = read(&write_f32(&[0, 3], &[])).unwrap();
        assert_eq!(arr.shape, vec![0, 3]);
        assert_eq!(arr.data, NpyData::F32(vec![]));
        let arr = read(&write_f32(&[], &[2.5])).unwrap();
        assert_eq!(arr.shape, Vec::<usize>::new());
        assert_eq!(arr.data, NpyData::F32(vec![2.5]));
    }

    #[test]
    #[should_panic(expected = "does not match data length")]
    fn npy_write_rejects_wrong_length() {
        let _ = write_f32(&[2, 3], &[0.0; 5]);
    }

    #[test]
    fn npy_reads_each_supported_dtype() {
        let f8: Vec<f64> = vec![1.5, -2.0, 3.25, 0.5, -0.125, 1e10];
        let payload: Vec<u8> = f8.iter().flat_map(|v| v.to_le_bytes()).collect();
        let arr = read(&file(&dict("<f8", "(2, 3)"), &payload)).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![2, 3], NpyData::F64(f8)));

        let arr = read(&file(
            &dict("<f4", "(3,)"),
            &[0, 0, 0xc0, 0x3f, 0, 0, 0x20, 0xc0, 0, 0, 0x80, 0x7f],
        ))
        .unwrap();
        assert_eq!(arr.data, NpyData::F32(vec![1.5, -2.5, f32::INFINITY]));

        let arr = read(&file(&dict("|b1", "(2, 2)"), &[1, 0, 0, 1])).unwrap();
        assert_eq!(
            (arr.shape, arr.data),
            (vec![2, 2], NpyData::Bool(vec![true, false, false, true]))
        );

        let arr = read(&file(&dict("|u1", "(4,)"), &[0, 1, 127, 255])).unwrap();
        assert_eq!(
            (arr.shape, arr.data),
            (vec![4], NpyData::U8(vec![0, 1, 127, 255]))
        );

        let arr = read(&file(
            &dict("<u2", "(2, 2)"),
            &[0, 0, 1, 0, 0, 1, 0xff, 0xff],
        ))
        .unwrap();
        assert_eq!(
            (arr.shape, arr.data),
            (vec![2, 2], NpyData::U16(vec![0, 1, 256, 65535]))
        );

        let arr = read(&file(&dict("<f8", "()"), &2.5f64.to_le_bytes())).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![], NpyData::F64(vec![2.5])));
    }

    #[test]
    fn npy_reads_version_2_header() {
        let d = dict("|u1", "(2,)");
        let mut text = d.into_bytes();
        text.resize(117, b' ');
        text.push(b'\n');
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&[2, 0]);
        bytes.extend_from_slice(&u32::try_from(text.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&text);
        bytes.extend_from_slice(&[7, 9]);
        let arr = read(&bytes).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![2], NpyData::U8(vec![7, 9])));
    }

    #[test]
    fn npy_rejects_unsupported_dtypes() {
        for descr in [">f8", "<i4", "<f2", ">u2", "<i8", "=f8", "|S3", "<c16"] {
            let err = read(&file(&dict(descr, "(1,)"), &[0; 16])).unwrap_err();
            assert_eq!(
                err,
                NpyError::UnsupportedDtype(descr.to_string()),
                "{descr}"
            );
        }
        let structured = "{'descr': [('a', '<f8')], 'fortran_order': False, 'shape': (1,), }";
        assert!(matches!(
            read(&file(structured, &[0; 8])),
            Err(NpyError::UnsupportedDtype(_))
        ));
    }

    /// `numpy.save(f, np.asfortranarray(np.arange(6, dtype='<u1').reshape(2, 3)))` (numpy 2.5.4).
    const NUMPY_FORTRAN_U1: &str = "934e554d5059010076007b276465736372273a20277c7531272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028322c2033292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000301040205";
    /// The same for `np.arange(12, dtype='<f4').reshape(2, 3, 2)`.
    const NUMPY_FORTRAN_F4: &str = "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028322c20332c2032292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a000000000000c040000000400000004100008040000020410000803f0000e04000004040000010410000a04000003041";
    /// The same for `np.asfortranarray(np.array([[True, False, False], [False, True, True]]))`.
    const NUMPY_FORTRAN_B1: &str = "934e554d5059010076007b276465736372273a20277c6231272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028322c2033292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a010000010001";

    #[test]
    fn npy_reads_fortran_order_arrays_in_c_order() {
        // the elements, as numpy shows them, are 0..6 in C order whatever the file order
        let arr = read(&hex(NUMPY_FORTRAN_U1)).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        assert_eq!(arr.data, NpyData::U8(vec![0, 1, 2, 3, 4, 5]));

        let arr = read(&hex(NUMPY_FORTRAN_F4)).unwrap();
        assert_eq!(arr.shape, vec![2, 3, 2]);
        let want: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(arr.data, NpyData::F32(want));

        let arr = read(&hex(NUMPY_FORTRAN_B1)).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
        let (t, f) = (true, false);
        assert_eq!(arr.data, NpyData::Bool(vec![t, f, f, f, t, t]));
    }

    #[test]
    fn npy_fortran_order_of_one_dimension_or_no_elements_is_the_same_as_c_order() {
        let d =
            |shape: &str| format!("{{'descr': '|u1', 'fortran_order': True, 'shape': {shape}, }}");
        let arr = read(&file(&d("(4,)"), &[9, 8, 7, 6])).unwrap();
        assert_eq!(arr.data, NpyData::U8(vec![9, 8, 7, 6]));
        let arr = read(&file(&d("()"), &[5])).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![], NpyData::U8(vec![5])));
        let arr = read(&file(&d("(0, 3)"), &[])).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![0, 3], NpyData::U8(vec![])));
        // a Fortran-order file is checked for its size like any other
        assert_eq!(
            read(&file(&d("(2, 3)"), &[0; 5])).unwrap_err(),
            NpyError::SizeMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    #[test]
    fn npy_rejects_bad_framing() {
        assert_eq!(read(b"").unwrap_err(), NpyError::BadMagic);
        assert_eq!(
            read(b"not an npy file at all").unwrap_err(),
            NpyError::BadMagic
        );
        assert_eq!(read(b"\x93NUMPY").unwrap_err(), NpyError::Truncated);
        assert_eq!(
            read(b"\x93NUMPY\x01\x00\x10").unwrap_err(),
            NpyError::Truncated
        );
        assert_eq!(
            read(b"\x93NUMPY\x04\x00\x10\x00").unwrap_err(),
            NpyError::UnsupportedVersion(4, 0)
        );
        assert_eq!(
            read(b"\x93NUMPY\x01\x01\x00\x00").unwrap_err(),
            NpyError::UnsupportedVersion(1, 1)
        );
        // header length points past the end of the data
        assert_eq!(
            read(b"\x93NUMPY\x01\x00\xff\xff{}").unwrap_err(),
            NpyError::Truncated
        );
        // header length overflowing the 32-bit field arithmetic
        assert_eq!(
            read(b"\x93NUMPY\x02\x00\xff\xff\xff\xff{}").unwrap_err(),
            NpyError::Truncated
        );
    }

    #[test]
    fn npy_rejects_bad_headers() {
        let cases = [
            "descr: '<f8'",
            "{'descr': '<f8', 'fortran_order': False}",
            "{'descr': '<f8', 'shape': (1,), }",
            "{'fortran_order': False, 'shape': (1,), }",
            "{'descr': '<f8', 'fortran_order': Maybe, 'shape': (1,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'extra': 1, }",
            "{'descr': '<f8', 'descr': '<f8', 'fortran_order': False, 'shape': (1,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (a,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': 3, }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,, }",
            "{'descr': '<f8' 'fortran_order': False, 'shape': (1,), }",
            "{'descr': 5, 'fortran_order': False, 'shape': (1,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,",
        ];
        for d in cases {
            let err = read(&file(d, &[0; 8])).unwrap_err();
            assert!(matches!(err, NpyError::BadHeader(_)), "{d}: {err:?}");
        }
    }

    /// A known key with the wrong kind of value is reported as that, not as an unknown key.
    #[test]
    fn npy_names_a_known_key_whose_value_has_the_wrong_type() {
        let cases = [
            (
                "{'descr': 5, 'fortran_order': False, 'shape': (1,), }",
                "descr",
            ),
            (
                "{'descr': '<f8', 'fortran_order': 'x', 'shape': (1,), }",
                "fortran_order",
            ),
            (
                "{'descr': '<f8', 'fortran_order': False, 'shape': 3, }",
                "shape",
            ),
        ];
        for (d, key) in cases {
            let NpyError::BadHeader(msg) = read(&file(d, &[0; 8])).unwrap_err() else {
                panic!("{d}: not a BadHeader");
            };
            assert!(msg.contains("unexpected value type"), "{d}: {msg}");
            assert!(msg.contains(key), "{d}: {msg}");
        }
        let NpyError::BadHeader(msg) = read(&file(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), 'extra': 1, }",
            &[0; 8],
        ))
        .unwrap_err() else {
            panic!("not a BadHeader");
        };
        assert!(msg.contains("unknown key"), "{msg}");
    }

    #[test]
    fn npy_rejects_size_mismatch() {
        let d = dict("<f8", "(2, 3)");
        assert_eq!(
            read(&file(&d, &[0; 40])).unwrap_err(),
            NpyError::SizeMismatch {
                expected: 48,
                actual: 40
            }
        );
        assert_eq!(
            read(&file(&d, &[0; 56])).unwrap_err(),
            NpyError::SizeMismatch {
                expected: 48,
                actual: 56
            }
        );
        let huge = dict("<f8", "(4294967296, 4294967296, 4294967296)");
        assert!(matches!(
            read(&file(&huge, &[])),
            Err(NpyError::BadHeader(_))
        ));
    }

    /// A shape whose element count and byte size fit `usize` but are far beyond the payload is a
    /// size mismatch: the payload is checked before anything is allocated for the elements.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn npy_huge_shape_with_a_short_payload_is_an_error_without_allocating() {
        for (descr, shape, expected) in [
            ("<f8", "(1099511627776,)", 8usize << 40),
            ("|u1", "(4294967295, 4294967295)", 4_294_967_295usize.pow(2)),
            ("<u2", "(9223372036854775807,)", usize::MAX - 1),
            ("|b1", "(18446744073709551615,)", usize::MAX),
        ] {
            let d = dict(descr, shape);
            for payload in [&[][..], &[1u8; 16][..]] {
                assert_eq!(
                    read(&file(&d, payload)).unwrap_err(),
                    NpyError::SizeMismatch {
                        expected,
                        actual: payload.len()
                    },
                    "{descr} {shape}"
                );
            }
            // in Fortran order as well
            let d = d.replace("False", "True");
            assert_eq!(
                read(&file(&d, &[1; 16])).unwrap_err(),
                NpyError::SizeMismatch {
                    expected,
                    actual: 16
                },
                "{descr} {shape} (Fortran)"
            );
        }
        // the byte size overflows usize although the element count does not
        let d = dict("<f8", "(9223372036854775807,)");
        assert!(matches!(
            read(&file(&d, &[1; 16])),
            Err(NpyError::BadHeader(_))
        ));
    }

    #[test]
    fn npy_accepts_header_whitespace_variants() {
        let d = "{ 'descr' : '<u2' , 'fortran_order' : False , 'shape' : ( 2 , ) }";
        let arr = read(&file(d, &[1, 0, 2, 0])).unwrap();
        assert_eq!((arr.shape, arr.data), (vec![2], NpyData::U16(vec![1, 2])));
        let d = "{\"descr\": \"|u1\", \"fortran_order\": False, \"shape\": (1,)}";
        assert_eq!(read(&file(d, &[9])).unwrap().data, NpyData::U8(vec![9]));
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn header_is_64_aligned_and_fits() {
        let dict_len: usize = kani::any();
        kani::assume(dict_len < 1 << 16);
        let total = header_len(dict_len);
        assert!(total % 64 == 0);
        // at least one space and the newline after the dict, as numpy pads (1 to 64 spaces); the
        // bound `total > 10 + dict_len` above also holds for a header that has the newline only
        assert!(total >= 10 + dict_len + 2);
        assert!(total <= 10 + dict_len + 1 + 64);
    }
}
