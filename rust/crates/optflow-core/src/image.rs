//! The row-major [`Image`] that every stage of the pipeline works on (grey levels, displacement
//! and confidence maps, masks). Decoding image files and video and reading ROIs is the job of the
//! command-line crate (`optflow-cli`); `.npy` files are [`crate::npy`].

/// Row-major image: pixel `(x, y)` lives at `data[y * w + x]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    pub w: usize,
    pub h: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Image<T> {
    /// A `w` x `h` image filled with `T::default()`.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![T::default(); w * h],
        }
    }

    /// Wraps row-major `data`. Panics if `data.len() != w * h`.
    pub fn from_vec(w: usize, h: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            w * h,
            "Image::from_vec: {} values for a {w}x{h} image",
            data.len()
        );
        Self { w, h, data }
    }

    /// The value at column `x`, row `y`. Panics if the pixel is outside the image.
    pub fn get(&self, x: usize, y: usize) -> T {
        assert!(
            x < self.w && y < self.h,
            "pixel ({x}, {y}) outside {}x{}",
            self.w,
            self.h
        );
        self.data[y * self.w + x]
    }

    /// Stores `v` at column `x`, row `y`. Panics if the pixel is outside the image.
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        assert!(
            x < self.w && y < self.h,
            "pixel ({x}, {y}) outside {}x{}",
            self.w,
            self.h
        );
        self.data[y * self.w + x] = v;
    }
}

/// Deterministic pseudo-random values in `[-1, 1)` (an LCG on `seed`), for the unit tests of the
/// modules that need some noise.
#[cfg(test)]
pub(crate) fn pseudo_random_image(w: usize, h: usize, seed: u64) -> Image<f64> {
    let mut s = seed;
    let data = (0..w * h)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect();
    Image::from_vec(w, h, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_new_is_zeroed_and_indexes_row_major() {
        let mut im: Image<f32> = Image::new(3, 2);
        assert_eq!(im.data, vec![0.0; 6]);
        im.set(2, 1, 7.0);
        assert_eq!(im.data[5], 7.0);
        assert_eq!(im.get(2, 1), 7.0);
        assert_eq!(im.get(0, 0), 0.0);
    }

    #[test]
    fn image_from_vec_keeps_data() {
        let im = Image::from_vec(2, 2, vec![1u8, 2, 3, 4]);
        assert_eq!((im.w, im.h, im.get(1, 0), im.get(0, 1)), (2, 2, 2, 3));
    }

    #[test]
    #[should_panic(expected = "Image::from_vec")]
    fn image_from_vec_rejects_wrong_length() {
        let _ = Image::from_vec(2, 2, vec![0.0f32; 3]);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn image_get_rejects_x_past_width() {
        // x = w would silently alias the first pixel of the next row without the bound check
        let im: Image<f32> = Image::new(3, 3);
        let _ = im.get(3, 0);
    }
}
