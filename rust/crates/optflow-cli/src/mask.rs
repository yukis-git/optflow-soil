//! `--mask PATH`: an ROI given as an image instead of a polygon (`load_mask` of
//! `python/optflow_ref/io.py` is the specification).

use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use optflow_core::image::Image;
use optflow_core::npy::{self, NpyData, NpyError};

/// Python's `str` of a shape tuple: `(2, 3)`, `(5,)`, `()`.
fn shape_text(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// numpy's name of an npy dtype string, as far as it is common.
fn dtype_name(descr: &str) -> &str {
    match descr {
        "<f2" => "float16",
        "<f4" => "float32",
        "<f8" => "float64",
        "|i1" => "int8",
        "<i2" => "int16",
        "<i4" => "int32",
        "<i8" => "int64",
        "<u2" => "uint16",
        "<u4" => "uint32",
        "<u8" => "uint64",
        other => other,
    }
}

/// The ROI of the file `path`: a `.npy` (bool or uint8) or a PNG, nonzero = inside, in pixels of
/// the scaled image, whose size is `w` x `h`. A colour PNG is inside where any channel is nonzero
/// (alpha is ignored).
pub fn load_mask(path: &Path, w: usize, h: usize) -> Result<Image<bool>> {
    let shown = path.display();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let (shape, inside): (Vec<usize>, Vec<bool>) = match ext.as_str() {
        "npy" => {
            let bytes =
                fs::read(path).map_err(|e| anyhow!("cannot read mask file {shown}: {e}"))?;
            let a = match npy::read(&bytes) {
                Ok(a) => a,
                Err(NpyError::UnsupportedDtype(descr)) => {
                    bail!(
                        "mask file {shown} must hold bool or uint8, got {}",
                        dtype_name(&descr)
                    )
                }
                Err(e) => bail!("invalid mask file {shown}: {e}"),
            };
            let inside = match a.data {
                NpyData::Bool(v) => v,
                NpyData::U8(v) => v.into_iter().map(|b| b != 0).collect(),
                NpyData::F32(_) => bail!("mask file {shown} must hold bool or uint8, got float32"),
                NpyData::F64(_) => bail!("mask file {shown} must hold bool or uint8, got float64"),
                NpyData::U16(_) => bail!("mask file {shown} must hold bool or uint8, got uint16"),
            };
            (a.shape, inside)
        }
        "png" => {
            let img = image::open(path)
                .map_err(|e| anyhow!("cannot read mask file {shown}: {e}"))?
                .to_rgb16();
            let inside = img.pixels().map(|p| p.0.iter().any(|&c| c != 0)).collect();
            (vec![img.height() as usize, img.width() as usize], inside)
        }
        _ => bail!("mask file {shown} must be a .npy or .png file"),
    };
    if shape != [h, w] {
        bail!(
            "mask file {shown} has shape {}, expected {}",
            shape_text(&shape),
            shape_text(&[h, w])
        );
    }
    Ok(Image::from_vec(w, h, inside))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_print_like_python_tuples() {
        assert_eq!(shape_text(&[64, 96]), "(64, 96)");
        assert_eq!(shape_text(&[128]), "(128,)");
        assert_eq!(shape_text(&[]), "()");
        assert_eq!(shape_text(&[2, 3, 4]), "(2, 3, 4)");
    }

    #[test]
    fn a_fortran_order_npy_mask_is_read_as_numpy_shows_it() {
        // np.save(f, np.asfortranarray(np.array([[True, False, False], [False, True, True]])))
        let hex = "934e554d5059010076007b276465736372273a20277c6231272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a2028322c2033292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a010000010001";
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let dir = crate::unit_scratch("fortran_mask");
        let path = dir.join("m.npy");
        fs::write(&path, bytes).unwrap();
        let m = load_mask(&path, 3, 2).unwrap();
        let (t, f) = (true, false);
        assert_eq!(m.data, [t, f, f, f, t, t]);
        // the shape is the array's, not the file's layout: (2, 3) is not (3, 2)
        let e = load_mask(&path, 2, 3).unwrap_err().to_string();
        assert!(e.ends_with("has shape (2, 3), expected (3, 2)"), "{e}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dtype_names_are_numpys() {
        assert_eq!(dtype_name("<i8"), "int64");
        assert_eq!(dtype_name("<f4"), "float32");
        assert_eq!(dtype_name("<c8"), "<c8");
    }
}
