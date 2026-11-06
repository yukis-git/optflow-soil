//! Wrappers that queue the spline, blur and warp kernels (`shaders/spline.metal`) on a [`Batch`].
//! They only queue; the caller's `Gpu::batch` waits.

use bytemuck::{Pod, Zeroable};
use optflow_core::gpu_model::df64::Df64;
use optflow_core::interp::gauss_kernel;

use crate::device::{Batch, Buffer, Gpu};
use crate::GpuError;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefilterP {
    w: u32,
    h: u32,
    es: u32,
    planes: u32,
    z: f32,
    scale: f32,
}

/// Pole `sqrt(3) - 2` and the start factor of the anti-causal pass as f32
/// (`gpu_model::field`).
fn prefilter_params(w: usize, h: usize, es: usize, planes: usize) -> PrefilterP {
    let z = 3.0f64.sqrt() - 2.0;
    PrefilterP {
        w: w as u32,
        h: h as u32,
        es: es as u32,
        planes: planes as u32,
        z: z as f32,
        scale: (z / (z * z - 1.0)) as f32,
    }
}

fn bad(msg: &str) -> GpuError {
    GpuError::Command(msg.to_string())
}

/// `w * h` for non-zero dimensions whose element count and any index fit `u32` with room for the
/// widest use (8 bytes per element and 2 floats per complex element).
fn check_dims(what: &str, w: usize, h: usize) -> Result<usize, GpuError> {
    if w == 0 || h == 0 {
        return Err(bad(&format!("{what}: empty {w}x{h} image")));
    }
    let n = w
        .checked_mul(h)
        .filter(|&n| n <= (u32::MAX as usize) / 16)
        .ok_or_else(|| bad(&format!("{what}: {w}x{h} is too large")))?;
    Ok(n)
}

fn check_len(what: &str, buf: &Buffer, need: usize) -> Result<(), GpuError> {
    if buf.len() < need {
        return Err(bad(&format!(
            "{what}: buffer has {} bytes, {need} needed",
            buf.len()
        )));
    }
    Ok(())
}

const G1: (usize, usize, usize) = (64, 1, 1);
const G2: (usize, usize, usize) = (8, 8, 1);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DfPrefilterP {
    w: u32,
    h: u32,
    es: u32,
    planes: u32,
    z: [f32; 2],
    scale: [f32; 2],
    zk_hi: [f32; HORIZON],
    zk_lo: [f32; HORIZON],
    mul: f32,
    first: u32,
    nf: u32,
    pad: u32,
}

/// Terms summed to start the causal pass (`interp::HORIZON`).
const HORIZON: usize = 21;

/// `data` holds the f32 planes, `scratch` the Df64 intermediate of `dims` = (w, h, es, planes).
fn check_df(
    what: &str,
    data: &Buffer,
    scratch: &Buffer,
    (w, h, es, planes): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let n = check_dims(what, w, h)?;
    if planes == 0 {
        return Err(bad(&format!("{what}: no planes")));
    }
    let floats = n
        .checked_mul(es * planes)
        .filter(|&f| f <= (u32::MAX as usize) / 2)
        .ok_or_else(|| bad(&format!("{what}: {planes} planes of {w}x{h} are too large")))?;
    check_len(what, data, 4 * floats)?;
    check_len(&format!("{what} scratch"), scratch, 8 * floats)
}

fn df_params(dims: (usize, usize, usize, usize)) -> DfPrefilterP {
    let (w, h, es, planes) = dims;
    let z = 3.0f64.sqrt() - 2.0;
    let scale = z / (z * z - 1.0);
    let df = |v: f64| Df64::from_f64(v);
    let mut zk = [1.0f64; HORIZON];
    for k in 1..HORIZON {
        zk[k] = zk[k - 1] * z;
    }
    DfPrefilterP {
        w: w as u32,
        h: h as u32,
        es: es as u32,
        planes: planes as u32,
        z: [df(z).hi, df(z).lo],
        scale: [df(scale).hi, df(scale).lo],
        zk_hi: zk.map(|v| df(v).hi),
        zk_lo: zk.map(|v| df(v).lo),
        mul: 1.0,
        first: 0,
        nf: 0,
        pad: 0,
    }
}

/// The recursion of `BSpline::new` (f64 operations in the order of `interp::prefilter_line`)
/// in Df64: pass along y from `src` (`planes` planes of `w x h` elements of `es` floats) into
/// `scratch` (`8 * planes * w * h * es` bytes: hi plane, then lo plane), then pass along x in
/// `scratch` and one rounding to f32 into `dst`.
fn prefilter_df(
    b: &mut Batch,
    src: &mut Buffer,
    scratch: &mut Buffer,
    dims: (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let (w, _, es, planes) = dims;
    check_df("prefilter", src, scratch, dims)?;
    let p = df_params(dims);
    b.dispatch(
        "prefilter_df_y",
        &mut [src, scratch],
        bytemuck::bytes_of(&p),
        (w * es, planes, 1),
        G1,
    )
}

fn prefilter_df_x(
    b: &mut Batch,
    scratch: &mut Buffer,
    dst: &mut Buffer,
    dims: (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let (_, h, es, planes) = dims;
    check_df("prefilter", dst, scratch, dims)?;
    let p = df_params(dims);
    b.dispatch(
        "prefilter_df_x",
        &mut [scratch, dst],
        bytemuck::bytes_of(&p),
        (h * es, planes, 1),
        G1,
    )
}

/// B-spline coefficients, in place, of `count` complex planes (w x h, interleaved f32): the
/// recursion runs in Df64 (as `BSpline::new` does in f64) and is rounded to f32 once. `scratch`
/// is `16 * w * h * count` bytes.
// Not used by `GpuEngine` since rev 2 (the fused kernels replaced it); kept as a tested
// reference.
#[doc(hidden)]
pub fn prefilter_complex(
    b: &mut Batch,
    data: &mut Buffer,
    scratch: &mut Buffer,
    w: usize,
    h: usize,
    count: usize,
) -> Result<(), GpuError> {
    let dims = (w, h, 2, count);
    prefilter_df(b, data, scratch, dims)?;
    prefilter_df_x(b, scratch, data, dims)
}

/// [`scale`] by `s`, [`prefilter_complex`] and [`interleave_filters`] in two dispatches, with the
/// same results: the B-spline coefficients of `count` complex planes (`w x h`, the inverse
/// transforms of the filters `first ..` of `n_filters`) times `s`, written to the node layout
/// `out[(node * n_filters + first + j)]`. `planes` is read only; `scratch` is
/// `16 * w * h * count` bytes.
#[allow(clippy::too_many_arguments)]
pub fn response_splines(
    b: &mut Batch,
    planes: &mut Buffer,
    scratch: &mut Buffer,
    (w, h): (usize, usize),
    count: usize,
    s: f32,
    (first, n_filters): (usize, usize),
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let dims = (w, h, 2, count);
    check_df("response_splines", planes, scratch, dims)?;
    check_filters("response_splines", count, first, n_filters)?;
    let nodes = (w * h)
        .checked_mul(n_filters)
        .filter(|&e| e <= (u32::MAX as usize) / 2)
        .ok_or_else(|| bad("response_splines: output too large"))?;
    check_len("response_splines out", out, 8 * nodes)?;
    let mut p = df_params(dims);
    p.mul = s;
    p.first = first as u32;
    p.nf = n_filters as u32;
    b.dispatch(
        "prefilter_df_y",
        &mut [planes, &mut *scratch],
        bytemuck::bytes_of(&p),
        (w * 2, count, 1),
        G1,
    )?;
    b.dispatch(
        "prefilter_df_x_nodes",
        &mut [scratch, out],
        bytemuck::bytes_of(&p),
        (h * 2 * count, 1, 1),
        G1,
    )
}

/// Coefficients of a real f32 image (the second image), Df64 recursion rounded once.
/// `scratch` is `8 * w * h` bytes.
pub fn prefilter_real(
    b: &mut Batch,
    img: &mut Buffer,
    scratch: &mut Buffer,
    out: &mut Buffer,
    w: usize,
    h: usize,
) -> Result<(), GpuError> {
    let dims = (w, h, 1, 1);
    prefilter_df(b, img, scratch, dims)?;
    prefilter_df_x(b, scratch, out, dims)
}

/// `field::spline_coeffs_centred` on the GPU.
pub fn prefilter_field(
    b: &mut Batch,
    s: &mut Buffer,
    out: &mut Buffer,
    w: usize,
    h: usize,
) -> Result<(), GpuError> {
    let n = check_dims("prefilter_field", w, h)?;
    check_len("prefilter_field s", s, 4 * n)?;
    check_len("prefilter_field out", out, 4 * n)?;
    let p = prefilter_params(w, h, 1, 1);
    b.dispatch(
        "field_prefilter_y",
        &mut [&mut *s, &mut *out],
        bytemuck::bytes_of(&p),
        (w, 1, 1),
        G1,
    )?;
    b.dispatch(
        "field_prefilter_x",
        &mut [s, out],
        bytemuck::bytes_of(&p),
        (h, 1, 1),
        G1,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurP {
    w: u32,
    h: u32,
    taps: u32,
    pad: u32,
}

/// The taps `gauss_kernel(sigma)` as f32 in a buffer, for [`blur_field`].
pub fn blur_taps(gpu: &Gpu, sigma: f64) -> Result<Buffer, GpuError> {
    let k: Vec<f32> = gauss_kernel(sigma).iter().map(|&v| v as f32).collect();
    let mut buf = gpu.buffer(4 * k.len())?;
    buf.as_mut_slice::<f32>().copy_from_slice(&k);
    Ok(buf)
}

/// `field::blur_centred` on the GPU: `u` as Df64 pairs (hi and lo planes), `out` f32. `taps` is
/// [`blur_taps`] (its length gives the radius), `mid` is scratch of `8 w h` bytes (the Df64
/// result of the pass along y).
#[allow(clippy::too_many_arguments)]
pub fn blur_field(
    b: &mut Batch,
    u_hi: &mut Buffer,
    u_lo: &mut Buffer,
    taps: &mut Buffer,
    mid: &mut Buffer,
    out: &mut Buffer,
    w: usize,
    h: usize,
) -> Result<(), GpuError> {
    let n = check_dims("blur_field", w, h)?;
    check_len("blur_field u_hi", u_hi, 4 * n)?;
    check_len("blur_field u_lo", u_lo, 4 * n)?;
    check_len("blur_field mid", mid, 8 * n)?;
    check_len("blur_field out", out, 4 * n)?;
    if taps.is_empty() || !taps.len().is_multiple_of(4) || (taps.len() / 4).is_multiple_of(2) {
        return Err(bad("blur_field: taps must be an odd number of f32"));
    }
    let head = BlurP {
        w: w as u32,
        h: h as u32,
        taps: (taps.len() / 4) as u32,
        pad: 0,
    };
    let params = bytemuck::bytes_of(&head);
    b.dispatch(
        "blur_y",
        &mut [u_hi, u_lo, taps, &mut *mid],
        params,
        (w, h, 1),
        G2,
    )?;
    b.dispatch("blur_x", &mut [mid, taps, out], params, (w, h, 1), G2)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WarpP {
    w: u32,
    h: u32,
    w2: u32,
    h2: u32,
    shx: i32,
    shy: i32,
    sixth_hi: f32,
    sixth_lo: f32,
}

/// `field::warp_exact` on the GPU: the coefficients of image 2 (`w2` x `h2`) sampled at
/// `(x + shift.0 + sx, y + shift.1 + sy)` for the `w` x `h` grid, Df64 weights and sums, rounded
/// once to f32.
#[allow(clippy::too_many_arguments)]
pub fn warp(
    b: &mut Batch,
    coeffs2: &mut Buffer,
    (w2, h2): (usize, usize),
    sx: &mut Buffer,
    sy: &mut Buffer,
    (w, h): (usize, usize),
    shift: (i64, i64),
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("warp", w, h)?;
    let n2 = check_dims("warp coefficients", w2, h2)?;
    check_len("warp coeffs2", coeffs2, 4 * n2)?;
    check_len("warp sx", sx, 4 * n)?;
    check_len("warp sy", sy, 4 * n)?;
    check_len("warp out", out, 4 * n)?;
    let sixth = Df64::from_f64(1.0 / 6.0);
    let lim = 1i64 << 29;
    let p = WarpP {
        w: w as u32,
        h: h as u32,
        w2: w2 as u32,
        h2: h2 as u32,
        shx: shift.0.clamp(-lim, lim) as i32,
        shy: shift.1.clamp(-lim, lim) as i32,
        sixth_hi: sixth.hi,
        sixth_lo: sixth.lo,
    };
    b.dispatch(
        "warp_exact",
        &mut [coeffs2, sx, sy, out],
        bytemuck::bytes_of(&p),
        (w, h, 1),
        G2,
    )
}

// ---- reference phase, responses and accumulate (`shaders/phase.metal`) ----

/// Reflect padding of the bank (`gabor::PAD`).
pub use optflow_core::gabor::PAD;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PadP {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
}

/// Reflect-pads `img` (w x h) by `PAD` into complex `out` ((w + 2 PAD) x (h + 2 PAD), imaginary
/// part 0): `gabor::pad_reflect`.
pub fn pad_reflect_complex(
    b: &mut Batch,
    img: &mut Buffer,
    w: usize,
    h: usize,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    check_dims("pad_reflect_complex", w, h)?;
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let np = check_dims("pad_reflect_complex", pw, ph)?;
    check_len("pad_reflect_complex img", img, 4 * w * h)?;
    check_len("pad_reflect_complex out", out, 8 * np)?;
    let p = PadP {
        w: w as u32,
        h: h as u32,
        pw: pw as u32,
        ph: ph as u32,
    };
    b.dispatch(
        "pad_reflect_complex",
        &mut [img, out],
        bytemuck::bytes_of(&p),
        (pw, ph, 1),
        G2,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MulP {
    n: u32,
    count: u32,
    pw: u32,
    ph: u32,
    axis: u32,
    pad: [u32; 3],
}

fn check_mul(
    what: &str,
    spectrum: &Buffer,
    filters: &Buffer,
    out: &Buffer,
    n: usize,
    count: usize,
) -> Result<(), GpuError> {
    if n == 0 || count == 0 || n > (u32::MAX as usize) / 16 || count > 4096 {
        return Err(bad(&format!("{what}: bad size {n} x {count}")));
    }
    check_len(&format!("{what} spectrum"), spectrum, 8 * n)?;
    check_len(&format!("{what} filters"), filters, 4 * n * count)?;
    check_len(&format!("{what} out"), out, 8 * n * count)
}

/// `out[k] = spectrum * filters[k]` for `count` real filters of `n` values each (complex
/// spectrum of `n` values): the filtered spectra, unscaled.
pub fn filter_multiply(
    b: &mut Batch,
    spectrum: &mut Buffer,
    filters: &mut Buffer,
    n: usize,
    count: usize,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    check_mul("filter_multiply", spectrum, filters, out, n, count)?;
    let p = MulP {
        n: n as u32,
        count: count as u32,
        pw: 1,
        ph: 1,
        axis: 0,
        pad: [0; 3],
    };
    b.dispatch(
        "filter_multiply",
        &mut [spectrum, filters, out],
        bytemuck::bytes_of(&p),
        (n, count, 1),
        G2,
    )
}

/// [`filter_multiply`] times `i f` (the spectrum of the derivative): along x for `axis == 0`
/// with `f = fx[column]`, along y otherwise with `f = fy[row]`. `freqs` holds `bank.fx`
/// (`pw` values) then `bank.fy` (`ph` values); the filters are `pw x ph` each.
#[allow(clippy::too_many_arguments)]
// Not used by `GpuEngine` since rev 2 (the fused kernels replaced it); kept as a tested
// reference.
#[doc(hidden)]
pub fn filter_multiply_deriv(
    b: &mut Batch,
    spectrum: &mut Buffer,
    filters: &mut Buffer,
    freqs: &mut Buffer,
    axis: u32,
    pw: usize,
    ph: usize,
    count: usize,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("filter_multiply_deriv", pw, ph)?;
    check_mul("filter_multiply_deriv", spectrum, filters, out, n, count)?;
    check_len("filter_multiply_deriv freqs", freqs, 4 * (pw + ph))?;
    if axis > 1 {
        return Err(bad("filter_multiply_deriv: axis must be 0 (x) or 1 (y)"));
    }
    let p = MulP {
        n: n as u32,
        count: count as u32,
        pw: pw as u32,
        ph: ph as u32,
        axis,
        pad: [0; 3],
    };
    b.dispatch(
        "filter_multiply_deriv",
        &mut [spectrum, filters, freqs, out],
        bytemuck::bytes_of(&p),
        (n, count, 1),
        G2,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleP {
    n: u32,
    s: f32,
}

/// Multiplies the first `n` f32 values of `data` by `s` in place (complex data: twice the
/// number of elements).
// Not used by `GpuEngine` since rev 2 (the fused kernels replaced it); kept as a tested
// reference.
#[doc(hidden)]
pub fn scale(b: &mut Batch, data: &mut Buffer, n: usize, s: f32) -> Result<(), GpuError> {
    if n > u32::MAX as usize {
        return Err(bad("scale: too many values"));
    }
    check_len("scale", data, 4 * n)?;
    if n == 0 {
        return Ok(());
    }
    let p = ScaleP { n: n as u32, s };
    b.dispatch(
        "scale_inplace",
        &mut [data],
        bytemuck::bytes_of(&p),
        (n, 1, 1),
        G1,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefP {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
    count: u32,
    first: u32,
    nf: u32,
    pad: u32,
    eps_hi: f32,
    eps_lo: f32,
}

fn check_filters(what: &str, count: usize, first: usize, n_filters: usize) -> Result<(), GpuError> {
    if count == 0 || first.checked_add(count).is_none_or(|e| e > n_filters) || n_filters > 4096 {
        return Err(bad(&format!(
            "{what}: filters {first}..{} out of {n_filters}",
            first.saturating_add(count)
        )));
    }
    Ok(())
}

/// The RefPack entries `[r.re, r.im, gx, gy, amp]` of `count` filters (numbers `first ..` of
/// `n_filters`) from their inverse transforms `r`, `rx`, `ry` (padded planes, already scaled by
/// 1/(pw ph)); `out` is `w * h * n_filters * 5` f32. `reference_values` in Df64, rounded once.
#[allow(clippy::too_many_arguments)]
// Not used by `GpuEngine` since rev 2 (the fused kernels replaced it); kept as a tested
// reference.
#[doc(hidden)]
pub fn reference_pack(
    b: &mut Batch,
    r: &mut Buffer,
    rx: &mut Buffer,
    ry: &mut Buffer,
    w: usize,
    h: usize,
    count: usize,
    first: usize,
    n_filters: usize,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("reference_pack", w, h)?;
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let np = check_dims("reference_pack", pw, ph)?;
    check_filters("reference_pack", count, first, n_filters)?;
    for (name, buf) in [("r", &*r), ("rx", &*rx), ("ry", &*ry)] {
        check_len(&format!("reference_pack {name}"), buf, 8 * np * count)?;
    }
    let entries = n
        .checked_mul(n_filters * 5)
        .filter(|&e| e <= u32::MAX as usize)
        .ok_or_else(|| bad("reference_pack: output too large"))?;
    check_len("reference_pack out", out, 4 * entries)?;
    let eps = Df64::from_f64(1e-12);
    let p = RefP {
        w: w as u32,
        h: h as u32,
        pw: pw as u32,
        ph: ph as u32,
        count: count as u32,
        first: first as u32,
        nf: n_filters as u32,
        pad: 0,
        eps_hi: eps.hi,
        eps_lo: eps.lo,
    };
    b.dispatch(
        "reference_pack",
        &mut [r, rx, ry, out],
        bytemuck::bytes_of(&p),
        (w, h, 1),
        G2,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InterP {
    n: u32,
    count: u32,
    first: u32,
    nf: u32,
}

/// Node layout of the responses' spline coefficients: the `count` complex planes (`pw x ph`)
/// become `out[(y * pw + x) * n_filters + first + j]` (complex).
#[allow(clippy::too_many_arguments)]
// Not used by `GpuEngine` since rev 2 (the fused kernels replaced it); kept as a tested
// reference.
#[doc(hidden)]
pub fn interleave_filters(
    b: &mut Batch,
    planes: &mut Buffer,
    pw: usize,
    ph: usize,
    count: usize,
    first: usize,
    n_filters: usize,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("interleave_filters", pw, ph)?;
    check_filters("interleave_filters", count, first, n_filters)?;
    check_len("interleave_filters planes", planes, 8 * n * count)?;
    let nodes = n
        .checked_mul(n_filters)
        .filter(|&e| e <= (u32::MAX as usize) / 2)
        .ok_or_else(|| bad("interleave_filters: output too large"))?;
    check_len("interleave_filters out", out, 8 * nodes)?;
    let p = InterP {
        n: n as u32,
        count: count as u32,
        first: first as u32,
        nf: n_filters as u32,
    };
    b.dispatch(
        "interleave_filters",
        &mut [planes, out],
        bytemuck::bytes_of(&p),
        (n, count, 1),
        G2,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccP {
    w: u32,
    h: u32,
    pw: u32,
    ph: u32,
    nf: u32,
    pad: [u32; 3],
}

/// The anchor-free per-pixel sums of `gpu_model::solve::pixel_sums`: `sums` is 7 f32 planes of
/// `w * h` (`gxx, gxy, gyy, hx, hy, s_phiphi, s_w`), NaN where the sampling coordinate is not
/// finite. `refpack` is the RefPack layout, `responses` the node layout (see
/// [`interleave_filters`]); `sx`, `sy` the smoothed field, `csx`, `csy` its centred spline
/// coefficients (all f32 `w x h`), `u*_hi/lo` the displacement as Df64 planes.
#[allow(clippy::too_many_arguments)]
pub fn accumulate(
    b: &mut Batch,
    refpack: &mut Buffer,
    responses: &mut Buffer,
    n_filters: usize,
    sx: &mut Buffer,
    sy: &mut Buffer,
    csx: &mut Buffer,
    csy: &mut Buffer,
    ux_hi: &mut Buffer,
    ux_lo: &mut Buffer,
    uy_hi: &mut Buffer,
    uy_lo: &mut Buffer,
    w: usize,
    h: usize,
    sums: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("accumulate", w, h)?;
    let (pw, ph) = (w + 2 * PAD, h + 2 * PAD);
    let np = check_dims("accumulate", pw, ph)?;
    check_filters("accumulate", 1, 0, n_filters)?;
    let pack = n
        .checked_mul(n_filters * 5)
        .filter(|&e| e <= u32::MAX as usize / 2)
        .ok_or_else(|| bad("accumulate: reference pack too large"))?;
    check_len("accumulate refpack", refpack, 4 * pack)?;
    let nodes = np
        .checked_mul(n_filters)
        .filter(|&e| e <= u32::MAX as usize / 2)
        .ok_or_else(|| bad("accumulate: responses too large"))?;
    check_len("accumulate responses", responses, 8 * nodes)?;
    for (name, buf) in [
        ("sx", &*sx),
        ("sy", &*sy),
        ("csx", &*csx),
        ("csy", &*csy),
        ("ux_hi", &*ux_hi),
        ("ux_lo", &*ux_lo),
        ("uy_hi", &*uy_hi),
        ("uy_lo", &*uy_lo),
    ] {
        check_len(&format!("accumulate {name}"), buf, 4 * n)?;
    }
    check_len("accumulate sums", sums, 28 * n)?;
    let p = AccP {
        w: w as u32,
        h: h as u32,
        pw: pw as u32,
        ph: ph as u32,
        nf: n_filters as u32,
        pad: [0; 3],
    };
    b.dispatch(
        "accumulate",
        &mut [
            refpack, responses, sx, sy, csx, csy, ux_hi, ux_lo, uy_hi, uy_lo, sums,
        ],
        bytemuck::bytes_of(&p),
        (w, h, 1),
        G2,
    )
}

// ---- window moments, the block-anchored local solve, the median reset (`shaders/moments.metal`) ----

use optflow_core::gpu_model::BLOCK;
use optflow_core::gpu_support::{moment_taps, plan_moments};

// the shader's threadgroup layout is written for 16 x 16 blocks
const _: () = assert!(BLOCK == 16);

/// Taps per moment power: powers 0 ..= 4 (the largest sum of two shape powers of order 2).
const MOMENT_POWERS: usize = 5;

/// Window radius of `sigma` and the device table `taps[p * (2 r + 1) + t]` of the f32 moment
/// kernels `k[t] (t - r)^p`, `p = 0 ..= 4` (`gpu_model::solve`).
pub fn moment_taps_buffer(gpu: &Gpu, sigma: f64) -> Result<(Buffer, usize), GpuError> {
    if !(sigma.is_finite() && sigma > 0.0 && sigma <= 256.0) {
        return Err(bad("moment_taps_buffer: sigma must be in (0, 256]"));
    }
    let taps = moment_taps(sigma, (MOMENT_POWERS - 1) as u32);
    let r = taps[0].len() / 2;
    let flat: Vec<f32> = taps.iter().flatten().map(|&v| v as f32).collect();
    let mut buf = gpu.buffer(4 * flat.len())?;
    buf.as_mut_slice::<f32>().copy_from_slice(&flat);
    Ok((buf, r))
}

/// The index tables of `plan_moments(order)` for the kernels, in a device buffer.
pub struct MomentPlanGpu {
    /// Unknowns of the shape function (6 or 12).
    pub n: usize,
    /// Window moments of the anchor-free sources Gxx, Gxy, Gyy, Gw: the planes of `global`.
    pub n_global: usize,
    table: Buffer,
    /// Planes of the y pass of the global moments (one per source and y power).
    pub n_yplanes: usize,
    /// Per y plane: source plane, py, slot count, then (slot, px) per slot (`MY_STRIDE` words).
    ytab: Buffer,
    /// Per global slot: its y plane and px.
    ztab: Buffer,
    /// The y-pass planes (`n_yplanes` planes of the image), grown to the image on use.
    yscratch: Option<Buffer>,
    /// The local moments (`n + 1` planes) and the block anchors (8 values per block) between
    /// `local_moments` and `solve_pixels`, grown on use.
    lscratch: Option<Buffer>,
    anchors: Option<Buffer>,
    /// Combos (local source, py) of the local moments' y pass, and that y pass per block
    /// (`local_y`), grown on use.
    n_combos: usize,
    ylocal: Option<Buffer>,
}

/// Window radius up to which the local moments take the three-kernel path (`LMAXR` of the
/// shader), and the row stride of its y-pass buffer (`LEW`).
const LMAXR: usize = 24;
const LEW: usize = BLOCK + 2 * LMAXR;
/// Most combos of the local moments' y pass (order 2), and the blocks that fit the grid's z
/// extent of `local_y` / `local_x`.
pub const MAX_LOCAL_COMBOS: usize = 7;
const MAX_Z_BLOCKS: usize = 65535;

/// Bytes of the local moments' y pass for `blocks` blocks of `n_combos` combos (with a block of
/// slack).
pub fn local_y_bytes(blocks: usize, n_combos: usize) -> usize {
    4 * (blocks * n_combos * BLOCK * LEW + BLOCK)
}

/// `slot` holding at least `bytes` bytes (a pooled buffer of the batch's device).
fn grown<'a>(
    b: &Batch,
    slot: &'a mut Option<Buffer>,
    bytes: usize,
) -> Result<&'a mut Buffer, GpuError> {
    if slot.as_ref().is_none_or(|y| y.len() < bytes) {
        if let Some(old) = slot.take() {
            b.gpu().release(old);
        }
        *slot = Some(b.gpu().buffer(bytes)?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

const PLAN_HEADER: usize = 20;
/// Words per y plane in the y-pass table (`MY_STRIDE` of the shader): at most 5 slots.
const MY_STRIDE: usize = 3 + 2 * 5;
/// Outputs per thread of the y and x passes of the global moments (`MY_ROWS` of the shader).
const MY_ROWS: usize = 16;
const GLOBAL_SOURCES: [u8; 4] = [0, 1, 2, 6];

impl MomentPlanGpu {
    pub fn new(gpu: &Gpu, order: u32) -> Result<Self, GpuError> {
        if order != 1 && order != 2 {
            return Err(bad("MomentPlanGpu: shape order must be 1 or 2"));
        }
        let (n, moments, m_idx, b_idx, gcc_idx, gw_idx) = plan_moments(order);
        // global slots: grouped by source, plan order within a source
        let mut slot = vec![u32::MAX; moments.len()];
        let mut gmom: Vec<u32> = Vec::new();
        let mut start = [0u32; 5];
        let mut maxpy = [0u32; 4];
        for (si, &src) in GLOBAL_SOURCES.iter().enumerate() {
            start[si] = (gmom.len() / 2) as u32;
            for (i, m) in moments.iter().enumerate().filter(|(_, m)| m.0 == src) {
                slot[i] = (gmom.len() / 2) as u32;
                gmom.extend([m.1 .0, m.1 .1]);
                maxpy[si] = maxpy[si].max(m.1 .1);
            }
        }
        let n_global = gmom.len() / 2;
        start[4] = n_global as u32;
        let mslot: Vec<u32> = m_idx.iter().map(|&i| slot[i]).collect();
        // local sources 3, 4, 5 -> 0, 1, 2; combos (source, py) (5 and 7 lmom entries n + 1 for
        // orders 1 and 2: the shader's local_moments_1/_2)
        let mut combos: Vec<(u32, u32)> = Vec::new();
        let mut lmom: Vec<u32> = Vec::new();
        for &i in b_idx.iter().chain([&gcc_idx]) {
            let (src, (px, py)) = moments[i];
            let key = (u32::from(src) - 3, py);
            let c = combos.iter().position(|&k| k == key).unwrap_or_else(|| {
                combos.push(key);
                combos.len() - 1
            });
            lmom.extend([c as u32, px]);
        }
        let want = if order == 1 { (5, 6) } else { (7, 12) };
        if (combos.len(), n) != want || lmom.len() != 2 * (n + 1) {
            return Err(bad(
                "MomentPlanGpu: the plan does not match the shader's layout",
            ));
        }
        let off_gmom = PLAN_HEADER;
        let off_mslot = off_gmom + gmom.len();
        let off_combo = off_mslot + mslot.len();
        let off_lmom = off_combo + 2 * combos.len();
        let mut t = vec![0u32; PLAN_HEADER];
        t[..8].copy_from_slice(&[
            n as u32,
            n_global as u32,
            slot[gw_idx],
            combos.len() as u32,
            off_gmom as u32,
            off_mslot as u32,
            off_combo as u32,
            off_lmom as u32,
        ]);
        t[8..13].copy_from_slice(&start);
        t[13..17].copy_from_slice(&maxpy);
        t.extend(&gmom);
        t.extend(&mslot);
        t.extend(combos.iter().flat_map(|&(a, b)| [a, b]));
        t.extend(&lmom);
        let mut table = gpu.buffer(4 * t.len())?;
        table.as_mut_slice::<u32>().copy_from_slice(&t);
        // the y planes of the global moments: (source, py) in slot order
        let mut yt: Vec<u32> = Vec::new();
        let mut keys: Vec<(u32, u32)> = Vec::new();
        for (si, &src) in GLOBAL_SOURCES.iter().enumerate() {
            for k in start[si] as usize..start[si + 1] as usize {
                let (px, py) = (gmom[2 * k], gmom[2 * k + 1]);
                let key = (u32::from(src), py);
                let z = keys.iter().position(|&q| q == key).unwrap_or_else(|| {
                    keys.push(key);
                    yt.extend([u32::from(src), py, 0]);
                    yt.extend([0; MY_STRIDE - 3]);
                    keys.len() - 1
                });
                let e = &mut yt[z * MY_STRIDE..(z + 1) * MY_STRIDE];
                let c = e[2] as usize;
                if c == 5 {
                    return Err(bad("MomentPlanGpu: more than 5 moments share a y pass"));
                }
                e[3 + 2 * c] = k as u32;
                e[4 + 2 * c] = px;
                e[2] += 1;
            }
        }
        let mut ytab = gpu.buffer(4 * yt.len())?;
        ytab.as_mut_slice::<u32>().copy_from_slice(&yt);
        let mut zt = vec![0u32; 2 * n_global];
        for (z, e) in yt.chunks(MY_STRIDE).enumerate() {
            for c in 0..e[2] as usize {
                let slot = e[3 + 2 * c] as usize;
                zt[2 * slot] = z as u32;
                zt[2 * slot + 1] = e[4 + 2 * c];
            }
        }
        let mut ztab = gpu.buffer(4 * zt.len())?;
        ztab.as_mut_slice::<u32>().copy_from_slice(&zt);
        Ok(Self {
            n,
            n_global,
            table,
            n_yplanes: keys.len(),
            ytab,
            ztab,
            yscratch: None,
            lscratch: None,
            anchors: None,
            n_combos: combos.len(),
            ylocal: None,
        })
    }

    /// Hands the plan's buffers back to the pool.
    pub fn release(self, gpu: &Gpu) {
        for b in [
            Some(self.table),
            Some(self.ytab),
            Some(self.ztab),
            self.yscratch,
            self.lscratch,
            self.anchors,
            self.ylocal,
        ]
        .into_iter()
        .flatten()
        {
            gpu.release(b);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalP {
    w: u32,
    h: u32,
    r: u32,
    ntaps: u32,
}

fn check_taps(what: &str, taps: &Buffer, r: usize) -> Result<(), GpuError> {
    if r > 4096 {
        return Err(bad(&format!("{what}: window radius {r} is too large")));
    }
    check_len(
        &format!("{what} taps"),
        taps,
        4 * MOMENT_POWERS * (2 * r + 1),
    )
}

/// Global window moments of the anchor-free sources Gxx, Gxy, Gyy, Gw (f32, separable, edges
/// replicated): `out[slot * w * h + i]` for the `plan.n_global` moments. `sums` are the 7
/// planes of [`accumulate`].
#[allow(clippy::too_many_arguments)]
pub fn moments_global(
    b: &mut Batch,
    sums: &mut Buffer,
    w: usize,
    h: usize,
    taps: &mut Buffer,
    r: usize,
    plan: &mut MomentPlanGpu,
    out: &mut Buffer,
) -> Result<(), GpuError> {
    let n = check_dims("moments_global", w, h)?;
    check_taps("moments_global", taps, r)?;
    check_len("moments_global sums", sums, 28 * n)?;
    let planes = n
        .checked_mul(plan.n_global)
        .filter(|&e| e <= u32::MAX as usize / 2)
        .ok_or_else(|| bad("moments_global: output too large"))?;
    check_len("moments_global out", out, 4 * planes)?;
    let p = GlobalP {
        w: w as u32,
        h: h as u32,
        r: r as u32,
        ntaps: (2 * r + 1) as u32,
    };
    let need = n
        .checked_mul(4 * plan.n_yplanes)
        .ok_or_else(|| bad("moments_global: y planes too large"))?;
    let MomentPlanGpu {
        n_global,
        n_yplanes,
        ytab,
        ztab,
        yscratch,
        ..
    } = plan;
    let ys = grown(b, yscratch, need)?;
    let nz = *n_yplanes;
    b.dispatch(
        "moments_global_y",
        &mut [sums, taps, ytab, ys],
        bytemuck::bytes_of(&p),
        (w.next_multiple_of(64), h.div_ceil(MY_ROWS), nz),
        (64, 1, 1),
    )?;
    b.dispatch(
        "moments_global_x",
        &mut [ys, taps, ztab, out],
        bytemuck::bytes_of(&p),
        (h.next_multiple_of(64), w.div_ceil(MY_ROWS), *n_global),
        (64, 1, 1),
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolveP {
    w: u32,
    h: u32,
    r: u32,
    ntaps: u32,
    nbx: u32,
    order_n: u32,
    ridge: f32,
    tol: f32,
    pi: f32,
    aconf: f32,
    aridge: f32,
    tol2: f32,
}

/// The aperture prototypes of the solve (`optflow_core::phaseflow::Aperture`, off by default):
/// the settings and `phase_tol^2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveAperture {
    pub conf: f64,
    pub ridge: f64,
    pub tol2: f64,
}

/// Number of blocks of a `w x h` image: the length of `erange` is three values per block.
pub fn block_count(w: usize, h: usize) -> usize {
    w.div_ceil(BLOCK) * h.div_ceil(BLOCK)
}

/// Per block of `BLOCK x BLOCK` pixels: the anchor, the increment-form moments over the block
/// plus the window radius, and the solve of its pixels (`gpu_model::solve::local_solve_blocks`
/// before the median reset). `u_in`/`u_out` are `[ux hi, ux lo, uy hi, uy lo]`; `slope` has 4
/// planes; `erange` gets `(min, max, has_nan)` of the energy per block (row-major blocks,
/// `block_count` of them).
#[allow(clippy::too_many_arguments)]
pub fn solve_blocks(
    b: &mut Batch,
    sums: &mut Buffer,
    global: &mut Buffer,
    u_in: [&mut Buffer; 4],
    w: usize,
    h: usize,
    taps: &mut Buffer,
    r: usize,
    plan: &mut MomentPlanGpu,
    u_out: [&mut Buffer; 4],
    energy: &mut Buffer,
    resid: &mut Buffer,
    slope: &mut Buffer,
    erange: &mut Buffer,
) -> Result<(), GpuError> {
    let off = SolveAperture::default();
    let outs = (energy, resid, slope, erange);
    solve_blocks_ap(b, sums, global, u_in, w, h, taps, r, plan, u_out, outs, off)
}

/// [`solve_blocks`] with the aperture prototypes (`gpu_model::solve::local_solve_blocks_ap`).
#[allow(clippy::too_many_arguments)]
pub fn solve_blocks_ap(
    b: &mut Batch,
    sums: &mut Buffer,
    global: &mut Buffer,
    u_in: [&mut Buffer; 4],
    w: usize,
    h: usize,
    taps: &mut Buffer,
    r: usize,
    plan: &mut MomentPlanGpu,
    u_out: [&mut Buffer; 4],
    (energy, resid, slope, erange): (&mut Buffer, &mut Buffer, &mut Buffer, &mut Buffer),
    ap: SolveAperture,
) -> Result<(), GpuError> {
    let n = check_dims("solve_blocks", w, h)?;
    if r <= LMAXR && block_count(w, h) > MAX_Z_BLOCKS {
        return Err(bad(&format!(
            "solve_blocks: {} blocks exceed the {MAX_Z_BLOCKS} of the local kernels' grid",
            block_count(w, h)
        )));
    }
    check_taps("solve_blocks", taps, r)?;
    check_len("solve_blocks sums", sums, 28 * n)?;
    let planes = n
        .checked_mul(plan.n_global)
        .filter(|&e| e <= u32::MAX as usize / 2)
        .ok_or_else(|| bad("solve_blocks: global moments too large"))?;
    check_len("solve_blocks global", global, 4 * planes)?;
    for (name, buf) in [
        ("u_in[0]", &*u_in[0]),
        ("u_in[1]", &*u_in[1]),
        ("u_in[2]", &*u_in[2]),
        ("u_in[3]", &*u_in[3]),
        ("u_out[0]", &*u_out[0]),
        ("u_out[1]", &*u_out[1]),
        ("u_out[2]", &*u_out[2]),
        ("u_out[3]", &*u_out[3]),
        ("energy", &*energy),
        ("resid", &*resid),
    ] {
        check_len(&format!("solve_blocks {name}"), buf, 4 * n)?;
    }
    check_len("solve_blocks slope", slope, 16 * n)?;
    let blocks = block_count(w, h);
    check_len("solve_blocks erange", erange, 12 * blocks)?;
    let p = SolveP {
        w: w as u32,
        h: h as u32,
        r: r as u32,
        ntaps: (2 * r + 1) as u32,
        nbx: w.div_ceil(BLOCK) as u32,
        order_n: plan.n as u32,
        ridge: 1e-12f64 as f32,
        tol: 1e-3f64 as f32,
        pi: std::f64::consts::PI as f32,
        aconf: ap.conf as f32,
        aridge: ap.ridge as f32,
        tol2: ap.tol2 as f32,
    };
    let [a0, a1, a2, a3] = u_in;
    let [o0, o1, o2, o3] = u_out;
    let (local, solve) = match plan.n {
        6 => ("local_moments_1", "solve_pixels_1"),
        _ => ("local_moments_2", "solve_pixels_2"),
    };
    let MomentPlanGpu {
        n: un,
        table,
        lscratch,
        anchors,
        n_combos,
        ylocal,
        ..
    } = plan;
    let lm = grown(b, lscratch, 4 * n * (*un + 1))?;
    let an = grown(b, anchors, 32 * blocks)?;
    let grid = (w.div_ceil(BLOCK) * 256, h.div_ceil(BLOCK), 1);
    if r <= LMAXR {
        let yl = grown(b, ylocal, local_y_bytes(blocks, *n_combos))?;
        let ew = BLOCK + 2 * r;
        b.dispatch(
            "block_anchors",
            &mut [a0, a1, a2, a3, an],
            bytemuck::bytes_of(&p),
            (blocks, 1, 1),
            G1,
        )?;
        b.dispatch(
            "local_y",
            &mut [sums, a0, a1, a2, a3, taps, table, an, yl],
            bytemuck::bytes_of(&p),
            (ew, *n_combos, blocks),
            (ew, *n_combos, 1),
        )?;
        b.dispatch(
            "local_x",
            &mut [yl, taps, table, lm],
            bytemuck::bytes_of(&p),
            (BLOCK, *un + 1, blocks),
            (BLOCK, *un + 1, 1),
        )?;
    } else {
        b.dispatch(
            local,
            &mut [sums, a0, a1, a2, a3, taps, table, lm, an],
            bytemuck::bytes_of(&p),
            grid,
            (256, 1, 1),
        )?;
    }
    b.dispatch(
        solve,
        &mut [
            global, lm, an, a0, a1, a2, a3, table, o0, o1, o2, o3, energy, resid, slope, erange,
        ],
        bytemuck::bytes_of(&p),
        grid,
        (256, 1, 1),
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResetP {
    n: u32,
    threshold: f32,
    pi: f32,
    pad: f32,
}

/// The largest f32 not above `t` (NaN stays NaN): `f64::from(e) > t` for an f32 `e` is
/// `e > round_down(t)`.
fn round_down_f32(t: f64) -> f32 {
    let f = t as f32;
    if f64::from(f) <= t || t.is_nan() {
        return f;
    }
    // f rounded up: step to the previous f32 (towards -inf)
    if f == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = f.to_bits();
    f32::from_bits(if f > 0.0 { bits - 1 } else { bits + 1 })
}

/// Resets the pixels whose energy is not above `threshold` (the rare median path; true for a
/// NaN energy) to the input displacement, `pi` and zero slope. Reads `energy`, `u_in`; writes
/// `u_out`, `resid`, `slope` (4 planes) of those pixels only.
#[allow(clippy::too_many_arguments)]
pub fn reset_below(
    b: &mut Batch,
    energy: &mut Buffer,
    threshold: f64,
    u_in: [&mut Buffer; 4],
    u_out: [&mut Buffer; 4],
    resid: &mut Buffer,
    slope: &mut Buffer,
    n: usize,
) -> Result<(), GpuError> {
    if n == 0 || n > u32::MAX as usize / 16 {
        return Err(bad("reset_below: bad pixel count"));
    }
    check_len("reset_below energy", energy, 4 * n)?;
    for buf in u_in.iter().chain(u_out.iter()) {
        check_len("reset_below u", buf, 4 * n)?;
    }
    check_len("reset_below resid", resid, 4 * n)?;
    check_len("reset_below slope", slope, 16 * n)?;
    let p = ResetP {
        n: n as u32,
        threshold: round_down_f32(threshold),
        pi: std::f64::consts::PI as f32,
        pad: 0.0,
    };
    let [a0, a1, a2, a3] = u_in;
    let [o0, o1, o2, o3] = u_out;
    b.dispatch(
        "reset_below",
        &mut [energy, a0, a1, a2, a3, o0, o1, o2, o3, resid, slope],
        bytemuck::bytes_of(&p),
        (n, 1, 1),
        G1,
    )
}
