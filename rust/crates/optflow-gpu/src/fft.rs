//! Arbitrary-size f32 2D FFT on Metal: Stockham stages in threadgroup memory, one threadgroup
//! per line; radices 8, 4, 2 and odd primes up to 31 as direct stages, Bluestein (padded to a
//! power of two) for lengths with a larger prime factor. Twiddles and chirps are computed in f64
//! on the CPU and rounded to f32.
//!
//! Direct lengths go through the register kernels (`fft_rows_regs_*`: the butterflies of a
//! stage in registers, the first stage reading and the last writing device memory); columns of
//! up to 1024 values are done in tiles of 4 to 16 adjacent columns (`fft_cols_tiled_*`), so
//! that the reads are consecutive values. Bluestein lengths keep the `fft_lines_bluestein_*`
//! kernels.

use std::f64::consts::PI;
use std::sync::Mutex;

use crate::device::{Batch, Buffer, Gpu};
use crate::GpuError;

/// Longest line the kernels hold in threadgroup memory (32 KB of complex f32).
const MAX_LEN: usize = 4096;
const MAX_STAGES: usize = 12;
const MAX_RADIX: usize = 31;
/// Upper bound on the threads per threadgroup the kernels are asked for; the pipelines' own
/// `maxTotalThreadsPerThreadgroup` (which can be lower) decides, see [`LinePlan::limit`].
const MAX_THREADS: usize = 1024;

/// The fewest threads per threadgroup any of the pipelines `kernels` allows.
fn pipeline_limit(gpu: &Gpu, kernels: &[&'static str]) -> Result<usize, GpuError> {
    let mut limit = MAX_THREADS;
    for k in kernels {
        limit = limit.min(gpu.max_threads(k)?);
    }
    Ok(limit)
}

/// The kernel for a line of `len` complex values (the padded length for Bluestein): the one
/// with the least threadgroup memory that holds it, since that memory limits the threadgroups
/// resident per core.
pub fn kernel_name_for(len: usize, bluestein: bool) -> &'static str {
    match (bluestein, len) {
        (false, ..=1024) => "fft_lines_direct_1k",
        (false, ..=2048) => "fft_lines_direct_2k",
        (false, _) => "fft_lines_direct_4k",
        (true, ..=1024) => "fft_lines_bluestein_1k",
        (true, ..=2048) => "fft_lines_bluestein_2k",
        (true, _) => "fft_lines_bluestein_4k",
    }
}

/// The register kernels by the largest radix of the line, as `(rows 1k, 2k, 4k, tiled)`: the
/// register use of a kernel is that of its largest stage, so a kernel without the large stages
/// runs more threads at once. The three sets match the MAXR template argument of the kernels
/// in `fft.metal` (`fft_stage_any` compiles the stages with R <= MAXR: 8 for 2, 3, 4, 5, 7, 8;
/// 19 adds 11, 13, 17, 19; 31 adds 23, 29, 31); a stage above the kernel's MAXR would do nothing,
/// see [`regs_maxr`].
fn regs_kernels(max_radix: usize) -> [&'static str; 4] {
    match max_radix {
        0..=8 => [
            "fft_rows_regs_r8_1k",
            "fft_rows_regs_r8_2k",
            "fft_rows_regs_r8_4k",
            "fft_cols_tiled_r8",
        ],
        9..=19 => [
            "fft_rows_regs_r19_1k",
            "fft_rows_regs_r19_2k",
            "fft_rows_regs_r19_4k",
            "fft_cols_tiled_r19",
        ],
        _ => [
            "fft_rows_regs_r31_1k",
            "fft_rows_regs_r31_2k",
            "fft_rows_regs_r31_4k",
            "fft_cols_tiled_r31",
        ],
    }
}

/// The `fft_rows_regs_mul` kernels (rows 1k, 2k, 4k) of the same MAXR sets as [`regs_kernels`]:
/// the first pass of [`Fft2Plan::inverse_filtered`].
fn regs_mul_kernels(max_radix: usize) -> [&'static str; 3] {
    match max_radix {
        0..=8 => [
            "fft_rows_regs_mul_r8_1k",
            "fft_rows_regs_mul_r8_2k",
            "fft_rows_regs_mul_r8_4k",
        ],
        9..=19 => [
            "fft_rows_regs_mul_r19_1k",
            "fft_rows_regs_mul_r19_2k",
            "fft_rows_regs_mul_r19_4k",
        ],
        _ => [
            "fft_rows_regs_mul_r31_1k",
            "fft_rows_regs_mul_r31_2k",
            "fft_rows_regs_mul_r31_4k",
        ],
    }
}

/// The MAXR of the kernels [`regs_kernels`] picks for `max_radix`.
fn regs_maxr(max_radix: usize) -> usize {
    match max_radix {
        0..=8 => 8,
        9..=19 => 19,
        _ => 31,
    }
}

/// The register-butterfly line kernel for a direct line of `len` values whose largest radix is
/// `max_radix`; it takes any element stride, so it also does the columns read with a stride.
pub fn regs_kernel_name_for(len: usize, max_radix: usize) -> &'static str {
    let k = regs_kernels(max_radix);
    match len {
        0..=1024 => k[0],
        1025..=2048 => k[1],
        _ => k[2],
    }
}

/// [`regs_kernel_name_for`] for the first pass of [`Fft2Plan::inverse_filtered`], which reads
/// the line as the spectrum times a filter.
pub fn regs_mul_kernel_name_for(len: usize, max_radix: usize) -> &'static str {
    let k = regs_mul_kernels(max_radix);
    match len {
        0..=1024 => k[0],
        1025..=2048 => k[1],
        _ => k[2],
    }
}

/// [`kernel_name_for`] of a Bluestein line for the first pass of [`Fft2Plan::inverse_filtered`].
pub fn bluestein_mul_kernel_name_for(len: usize) -> &'static str {
    match len {
        0..=1024 => "fft_lines_bluestein_mul_1k",
        1025..=2048 => "fft_lines_bluestein_mul_2k",
        _ => "fft_lines_bluestein_mul_4k",
    }
}

/// The `fft_cols_tiled` variant for lines whose largest radix is `max_radix`.
pub fn tiled_kernel_name_for(max_radix: usize) -> &'static str {
    regs_kernels(max_radix)[3]
}

/// Radices `fft_line_regs` has a stage for (all that `factorise` gives for a direct length).
const REGS_RADICES: [usize; 13] = [2, 3, 4, 5, 7, 8, 11, 13, 17, 19, 23, 29, 31];

/// Complex values of the threadgroup memory of `fft_cols_tiled`.
const TILE_VALUES: usize = 4096;

/// How the columns of a [`Fft2Plan`] are transformed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnPass {
    /// One threadgroup per column, the column read with a stride of a row.
    Strided,
    /// `fft_cols_tiled`: one threadgroup per `c` adjacent columns, so that every read is `c`
    /// consecutive values.
    Tiled(usize),
    /// Transpose into a scratch plane, transform its rows, transpose back.
    Transposed,
}

/// How a length is transformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Factorisation {
    /// Radix sequence (product = n) of direct stages.
    Direct(Vec<usize>),
    /// Convolution of padded length `m` (a power of two >= 2n - 1).
    Bluestein { n: usize, m: usize },
}

fn pow2_radices(mut n: usize) -> Vec<usize> {
    let mut r = Vec::new();
    while n.is_multiple_of(8) {
        r.push(8);
        n /= 8;
    }
    if n.is_multiple_of(4) {
        r.push(4);
        n /= 4;
    }
    if n.is_multiple_of(2) {
        r.push(2);
    }
    r
}

/// The radix sequence used for a length: factors 8, 4, 2, 3, 5, 7 and odd primes up to 31 as
/// direct DFT stages; a prime factor above 31 makes the whole length go through Bluestein.
pub fn factorise(n: usize) -> Factorisation {
    assert!(n >= 1);
    let twos = n.trailing_zeros();
    let mut rest = n >> twos;
    let mut radices = pow2_radices(1 << twos);
    let mut p = 3;
    while p <= MAX_RADIX {
        while rest.is_multiple_of(p) {
            radices.push(p);
            rest /= p;
        }
        p += 2;
    }
    if rest == 1 {
        Factorisation::Direct(radices)
    } else {
        Factorisation::Bluestein {
            n,
            m: (2 * n - 1).next_power_of_two(),
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    nlen: u32,
    len: u32,
    nst: u32,
    estride: u32,
    lstride: u32,
    pstride: u32,
    nlines: u32,
    sgn: f32,
    /// the last write of the pass multiplies by it (1 leaves the values exact)
    scale: f32,
    r: [u32; MAX_STAGES],
}

/// Buffers a plan owns. The GPU only reads them; the mutex is there because `Batch::dispatch`
/// takes every buffer as `&mut`.
struct Table(Mutex<Buffer>);

impl Table {
    fn new(gpu: &Gpu, values: &[[f32; 2]]) -> Result<Table, GpuError> {
        let mut b = gpu.buffer(8 * values.len().max(1))?;
        for (dst, v) in b.as_mut_slice::<[f32; 2]>().iter_mut().zip(values) {
            *dst = *v;
        }
        Ok(Table(Mutex::new(b)))
    }
}

struct Bluestein {
    chirp: Table,
    forward_filter: Table,
    inverse_filter: Table,
}

/// The input of the first pass of [`Fft2Plan::inverse_filtered`]: one spectrum plane and the real
/// filters, one plane per output plane.
struct Product<'a> {
    spec: &'a mut Buffer,
    filters: &'a mut Buffer,
}

/// Where the lines of one axis sit in a batch of planes (in complex values).
#[derive(Clone, Copy)]
struct Layout {
    elem_stride: usize,
    line_stride: usize,
    lines: usize,
    plane_stride: usize,
}

/// The transform of one axis.
struct LinePlan {
    n: usize,
    /// transform length: `n`, or the padded length for Bluestein
    len: usize,
    radices: Vec<u32>,
    threads: usize,
    /// threads per threadgroup the pipeline of the tiled column kernel allows (0 when the line
    /// is not tileable)
    tile_limit: usize,
    /// direct with stages `fft_line_regs` has
    regs: bool,
    twiddles: Table,
    bluestein: Option<Bluestein>,
}

fn twiddle_table(len: usize) -> Vec<[f32; 2]> {
    (0..len)
        .map(|k| {
            let a = -2.0 * PI * k as f64 / len as f64;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect()
}

/// In-place forward radix-2 FFT in f64 (`a.len()` a power of two).
fn fft_f64(a: &mut [[f64; 2]]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        for start in (0..n).step_by(size) {
            for k in 0..half {
                let ang = -2.0 * PI * k as f64 / size as f64;
                let (s, c) = ang.sin_cos();
                let u = a[start + k];
                let x = a[start + k + half];
                let v = [x[0] * c - x[1] * s, x[0] * s + x[1] * c];
                a[start + k] = [u[0] + v[0], u[1] + v[1]];
                a[start + k + half] = [u[0] - v[0], u[1] - v[1]];
            }
        }
        size *= 2;
    }
}

impl LinePlan {
    fn new(gpu: &Gpu, n: usize) -> Result<LinePlan, GpuError> {
        let too_long = |what: &str, len: usize| {
            GpuError::Command(format!(
                "FFT length {n}: {what} {len} exceeds the {MAX_LEN} complex values a threadgroup holds"
            ))
        };
        match factorise(n) {
            Factorisation::Direct(radices) => {
                // every stage needs one thread per butterfly
                let smallest = radices.iter().copied().min().unwrap_or(1);
                let threads = (n / smallest).max(1);
                let regs = radices.iter().all(|r| REGS_RADICES.contains(r));
                if n > MAX_LEN {
                    return Err(too_long("line", n));
                }
                let (limit, tile_limit) = if regs {
                    let maxr = radices.iter().copied().max().unwrap_or(1);
                    let limit = pipeline_limit(
                        gpu,
                        &[
                            regs_kernel_name_for(n, maxr),
                            regs_mul_kernel_name_for(n, maxr),
                        ],
                    )?;
                    (limit, pipeline_limit(gpu, &[tiled_kernel_name_for(maxr)])?)
                } else {
                    (MAX_THREADS, 0)
                };
                if threads > limit {
                    return Err(GpuError::Command(format!(
                        "FFT length {n}: {threads} threads per line exceed the pipeline limit {limit}"
                    )));
                }
                Ok(LinePlan {
                    n,
                    len: n,
                    radices: radices.iter().map(|&r| r as u32).collect(),
                    threads,
                    tile_limit,
                    regs,
                    twiddles: Table::new(gpu, &twiddle_table(n))?,
                    bluestein: None,
                })
            }
            Factorisation::Bluestein { n, m } => {
                if m > MAX_LEN {
                    return Err(too_long("padded length", m));
                }
                let radices = pow2_radices(m);
                let smallest = radices.iter().copied().min().unwrap_or(1);
                let limit = pipeline_limit(
                    gpu,
                    &[kernel_name_for(m, true), bluestein_mul_kernel_name_for(m)],
                )?;
                // chirp c_k = exp(-pi i k^2 / n), k^2 reduced mod 2n in integers
                let chirp: Vec<[f64; 2]> = (0..n)
                    .map(|k| {
                        let k2 = ((k as u64 * k as u64) % (2 * n as u64)) as f64;
                        let a = -PI * k2 / n as f64;
                        [a.cos(), a.sin()]
                    })
                    .collect();
                // filter a_t = conj(c_t), wrapped to length m; for the inverse direction the
                // conjugate sequence. Both are transformed forward.
                let filter = |conj: bool| {
                    let mut f = vec![[0.0f64; 2]; m];
                    for t in 0..n {
                        let v = if conj {
                            chirp[t]
                        } else {
                            [chirp[t][0], -chirp[t][1]]
                        };
                        f[t] = v;
                        if t > 0 {
                            f[m - t] = v;
                        }
                    }
                    fft_f64(&mut f);
                    f.iter()
                        .map(|c| [c[0] as f32, c[1] as f32])
                        .collect::<Vec<_>>()
                };
                let chirp32: Vec<[f32; 2]> =
                    chirp.iter().map(|c| [c[0] as f32, c[1] as f32]).collect();
                Ok(LinePlan {
                    n,
                    len: m,
                    radices: radices.iter().map(|&r| r as u32).collect(),
                    threads: (m / smallest).min(limit),
                    tile_limit: 0,
                    regs: false,
                    twiddles: Table::new(gpu, &twiddle_table(m))?,
                    bluestein: Some(Bluestein {
                        chirp: Table::new(gpu, &chirp32)?,
                        forward_filter: Table::new(gpu, &filter(false))?,
                        inverse_filter: Table::new(gpu, &filter(true))?,
                    }),
                })
            }
        }
    }

    /// Every radix has a stage in the register kernel chosen for the line (`fft_stage_any`
    /// skips a radix it has no stage for).
    fn debug_assert_regs_stages(&self) {
        let maxr = regs_maxr(self.max_radix());
        debug_assert!(
            self.radices
                .iter()
                .all(|&r| REGS_RADICES.contains(&(r as usize)) && r as usize <= maxr),
            "radices {:?} not all in the MAXR {maxr} register kernel",
            self.radices
        );
    }

    fn max_radix(&self) -> usize {
        self.radices.iter().copied().max().unwrap_or(1) as usize
    }

    /// The pass of this axis can read the [`Product`] itself (register and Bluestein lines).
    fn can_fuse_product(&self) -> bool {
        self.n > 1 && (self.regs || self.bluestein.is_some())
    }

    fn params(&self, inverse: bool, lay: Layout, scale: f32) -> Params {
        let Layout {
            elem_stride,
            line_stride,
            lines,
            plane_stride,
        } = lay;
        let mut r = [0u32; MAX_STAGES];
        r[..self.radices.len()].copy_from_slice(&self.radices);
        Params {
            nlen: self.n as u32,
            len: self.len as u32,
            nst: self.radices.len() as u32,
            estride: elem_stride as u32,
            lstride: line_stride as u32,
            pstride: plane_stride as u32,
            nlines: lines as u32,
            sgn: if inverse { -1.0 } else { 1.0 },
            scale,
            r,
        }
    }

    /// The widest tile of adjacent lines `fft_cols_tiled` can take for this length (16, 8 or 4
    /// lines, within its threadgroup memory and the thread limit of its pipeline), if any. Tiles of 2 (lengths up
    /// to 2048) measured no faster than the strided columns on the M4 (spec §11).
    fn tile_lines(&self) -> Option<usize> {
        [16, 8, 4].into_iter().find(|&c| {
            self.regs && c * self.len <= TILE_VALUES && c * self.threads <= self.tile_limit
        })
    }

    /// Transform `lay.lines` lines per plane of `count` planes, `c` adjacent lines (line stride
    /// 1) per threadgroup with `fft_cols_tiled`; the output times `scale`.
    #[allow(clippy::too_many_arguments)]
    fn run_tiled(
        &self,
        b: &mut Batch,
        data: &mut Buffer,
        inverse: bool,
        lay: Layout,
        count: usize,
        c: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        if self.n == 1 {
            return Ok(());
        }
        if !self.regs
            || lay.line_stride != 1
            || c * self.len > TILE_VALUES
            || c * self.threads > self.tile_limit
        {
            return Err(GpuError::Command(format!(
                "fft_cols_tiled: {c} lines of length {} do not fit",
                self.len
            )));
        }
        self.debug_assert_regs_stages();
        let params = self.params(inverse, lay, scale);
        let mut tw = self.twiddles.0.lock().unwrap_or_else(|e| e.into_inner());
        b.dispatch(
            tiled_kernel_name_for(self.max_radix()),
            &mut [data, &mut tw],
            bytemuck::bytes_of(&params),
            (lay.lines.div_ceil(c) * c, self.threads, count),
            (c, self.threads, 1),
        )
    }

    /// Transform `lay.lines` lines per plane of `count` planes, the output times `scale`. With a
    /// `product` (which needs [`LinePlan::can_fuse_product`]) the lines are read from it, not
    /// from `data`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        b: &mut Batch,
        data: &mut Buffer,
        inverse: bool,
        lay: Layout,
        count: usize,
        scale: f32,
        product: Option<Product<'_>>,
    ) -> Result<(), GpuError> {
        if self.n == 1 {
            debug_assert!(product.is_none(), "a length-1 pass cannot read the product");
            return Ok(()); // the length-1 DFT is the identity
        }
        let params = self.params(inverse, lay, scale);
        let grid = (self.threads, lay.lines, count);
        let group = (self.threads, 1, 1);
        // every direct length factorises into register radices; only Bluestein has no `regs`
        if !self.regs && self.bluestein.is_none() {
            return Err(GpuError::Command(format!(
                "FFT length {}: a direct line without register stages has no kernel",
                self.n
            )));
        }
        if self.regs {
            self.debug_assert_regs_stages();
        }
        let mut tw = self.twiddles.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Product { spec, filters }) = product {
            return match &self.bluestein {
                None => b.dispatch(
                    regs_mul_kernel_name_for(self.len, self.max_radix()),
                    &mut [data, &mut tw, spec, filters],
                    bytemuck::bytes_of(&params),
                    grid,
                    group,
                ),
                Some(bl) => {
                    let mut chirp = bl.chirp.0.lock().unwrap_or_else(|e| e.into_inner());
                    let mut bf = bl
                        .forward_filter
                        .0
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let mut bi = bl
                        .inverse_filter
                        .0
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    b.dispatch(
                        bluestein_mul_kernel_name_for(self.len),
                        &mut [data, &mut tw, &mut chirp, &mut bf, &mut bi, spec, filters],
                        bytemuck::bytes_of(&params),
                        grid,
                        group,
                    )
                }
            };
        }
        match &self.bluestein {
            None => b.dispatch(
                regs_kernel_name_for(self.len, self.max_radix()),
                &mut [data, &mut tw],
                bytemuck::bytes_of(&params),
                grid,
                group,
            ),
            Some(bl) => {
                let mut chirp = bl.chirp.0.lock().unwrap_or_else(|e| e.into_inner());
                let mut bf = bl
                    .forward_filter
                    .0
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let mut bi = bl
                    .inverse_filter
                    .0
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                b.dispatch(
                    kernel_name_for(self.len, true),
                    &mut [data, &mut tw, &mut chirp, &mut bf, &mut bi],
                    bytemuck::bytes_of(&params),
                    grid,
                    group,
                )
            }
        }
    }
}

/// A 2D transform of `h` x `w` row-major complex planes.
pub struct Fft2Plan {
    h: usize,
    w: usize,
    rows: LinePlan,
    cols: LinePlan,
    column_pass: ColumnPass,
    /// the transposed planes of [`ColumnPass::Transposed`], grown to the largest batch
    scratch: Mutex<Option<Buffer>>,
}

impl Fft2Plan {
    pub fn new(gpu: &Gpu, h: usize, w: usize) -> Result<Fft2Plan, GpuError> {
        let mut plan = Fft2Plan::with_column_pass(gpu, h, w, ColumnPass::Strided)?;
        // the fastest choice measured on the M4 (spec §11): tiles of adjacent columns where 4 or
        // more fit, else the strided columns (the transposes cost about as much as the strided
        // column pass itself)
        if let Some(c) = plan.cols.tile_lines() {
            plan.column_pass = ColumnPass::Tiled(c);
        }
        Ok(plan)
    }

    /// A plan with the given column pass (the default is the fastest; this is for tests and
    /// benchmarks). `Tiled(c)` needs a direct column length with `c` columns fitting the tile.
    #[doc(hidden)]
    pub fn with_column_pass(
        gpu: &Gpu,
        h: usize,
        w: usize,
        column_pass: ColumnPass,
    ) -> Result<Fft2Plan, GpuError> {
        if h == 0 || w == 0 {
            return Err(GpuError::Command(format!("FFT of an empty {h}x{w} plane")));
        }
        let cols = LinePlan::new(gpu, h)?;
        if let ColumnPass::Tiled(c) = column_pass {
            if !cols.regs
                || c == 0
                || c * cols.len > TILE_VALUES
                || c * cols.threads > cols.tile_limit
            {
                return Err(GpuError::Command(format!(
                    "FFT columns of length {h}: tiles of {c} do not fit"
                )));
            }
        }
        Ok(Fft2Plan {
            h,
            w,
            rows: LinePlan::new(gpu, w)?,
            cols,
            column_pass,
            scratch: Mutex::new(None),
        })
    }

    /// The column pass the plan uses.
    pub fn column_pass(&self) -> ColumnPass {
        self.column_pass
    }

    /// `count` planes of `rows` x `cols` from `src` to their transposes in `dst`, times `scale`.
    fn transpose(
        b: &mut Batch,
        src: &mut Buffer,
        dst: &mut Buffer,
        rows: usize,
        cols: usize,
        count: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let params = Params {
            nlen: 0,
            len: 0,
            nst: 0,
            estride: cols as u32,
            lstride: rows as u32,
            pstride: (rows * cols) as u32,
            nlines: 0,
            sgn: 0.0,
            scale,
            r: [0; MAX_STAGES],
        };
        b.dispatch(
            "transpose_c32",
            &mut [src, dst],
            bytemuck::bytes_of(&params),
            (cols.div_ceil(32) * 32, rows.div_ceil(32) * 32, count),
            (32, 32, 1),
        )
    }

    /// Rows, then columns, the last pass that runs scaling by `scale` (none runs for a 1 x 1
    /// plane: then `scale` is not applied). With a `product` the row pass reads it (the rows
    /// need [`LinePlan::can_fuse_product`]).
    fn run(
        &self,
        b: &mut Batch,
        data: &mut Buffer,
        count: usize,
        inverse: bool,
        scale: f32,
        product: Option<Product<'_>>,
    ) -> Result<(), GpuError> {
        let plane = self.h * self.w;
        if data.len() < 8 * plane * count {
            return Err(GpuError::Command(format!(
                "FFT of {count} {}x{} planes needs {} bytes, the buffer has {}",
                self.h,
                self.w,
                8 * plane * count,
                data.len()
            )));
        }
        if count == 0 {
            return Ok(());
        }
        let rows = Layout {
            elem_stride: 1,
            line_stride: self.w,
            lines: self.h,
            plane_stride: plane,
        };
        let cols = Layout {
            elem_stride: self.w,
            line_stride: 1,
            lines: self.w,
            plane_stride: plane,
        };
        // the columns are the last pass unless they are of length 1
        let row_scale = if self.cols.n == 1 { scale } else { 1.0 };
        self.rows
            .run(b, data, inverse, rows, count, row_scale, product)?;
        match self.column_pass {
            ColumnPass::Strided => self.cols.run(b, data, inverse, cols, count, scale, None),
            ColumnPass::Tiled(c) => self.cols.run_tiled(b, data, inverse, cols, count, c, scale),
            ColumnPass::Transposed if self.cols.n == 1 => Ok(()),
            ColumnPass::Transposed => {
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                let bytes = 8 * plane * count;
                if scratch.as_ref().is_none_or(|s| s.len() < bytes) {
                    *scratch = Some(b.gpu().buffer(bytes)?);
                }
                let t = scratch.as_mut().expect("allocated above");
                Self::transpose(b, data, t, self.h, self.w, count, 1.0)?;
                let lines = Layout {
                    elem_stride: 1,
                    line_stride: self.h,
                    lines: self.w,
                    plane_stride: plane,
                };
                self.cols.run(b, t, inverse, lines, count, 1.0, None)?;
                Self::transpose(b, t, data, self.w, self.h, count, scale)
            }
        }
    }

    /// In place on `count` complex h x w planes stored back to back (interleaved re, im f32):
    /// forward unnormalised, like rustfft; rows first, then columns (as `optflow_core::fft::Fft2`).
    pub fn forward(&self, b: &mut Batch, data: &mut Buffer, count: usize) -> Result<(), GpuError> {
        self.run(b, data, count, false, 1.0, None)
    }

    /// Inverse, unnormalised (the caller scales by 1/(h w), like `Fft2::inverse_parts`).
    pub fn inverse(&self, b: &mut Batch, data: &mut Buffer, count: usize) -> Result<(), GpuError> {
        self.run(b, data, count, true, 1.0, None)
    }

    /// `out[k] = scale * inverse(spec * filters[k])` for `k < count`: `spec` one complex h x w
    /// plane, `filters` `count` real planes of `np = h w` values back to back, `out` `count`
    /// complex planes. The product is read by the row pass and `scale` applied by the last
    /// write, with the same f32 operations as `filter_multiply`, [`Fft2Plan::inverse`] and then
    /// `x * scale` on every value, so the result has the same bits. Rows that cannot read the
    /// product (a length 1) get it from a separate `fft_filter_multiply` pass.
    #[allow(clippy::too_many_arguments)]
    pub fn inverse_filtered(
        &self,
        b: &mut Batch,
        spec: &mut Buffer,
        filters: &mut Buffer,
        np: usize,
        count: usize,
        out: &mut Buffer,
        scale: f32,
    ) -> Result<(), GpuError> {
        let plane = self.h * self.w;
        if np != plane {
            return Err(GpuError::Command(format!(
                "inverse_filtered: {np} values per plane, the plan is {}x{}",
                self.h, self.w
            )));
        }
        let need = |what: &str, buf: &Buffer, bytes: usize| {
            if buf.len() < bytes {
                Err(GpuError::Command(format!(
                    "inverse_filtered: {what} needs {bytes} bytes, the buffer has {}",
                    buf.len()
                )))
            } else {
                Ok(())
            }
        };
        need("the spectrum", spec, 8 * plane)?;
        need("the filters", filters, 4 * plane * count)?;
        need("the output", out, 8 * plane * count)?;
        if count == 0 {
            return Ok(());
        }
        if self.rows.can_fuse_product() {
            return self.run(b, out, count, true, scale, Some(Product { spec, filters }));
        }
        // no pass runs for a 1 x 1 plane: the multiply then also scales
        let no_pass = self.rows.n == 1 && self.cols.n == 1;
        let params = Params {
            nlen: plane as u32,
            len: 0,
            nst: 0,
            estride: 0,
            lstride: 0,
            pstride: 0,
            nlines: count as u32,
            sgn: 0.0,
            scale: if no_pass { scale } else { 1.0 },
            r: [0; MAX_STAGES],
        };
        b.dispatch(
            "fft_filter_multiply",
            &mut [spec, filters, &mut *out],
            bytemuck::bytes_of(&params),
            (plane, count, 1),
            (64, 1, 1),
        )?;
        self.run(b, out, count, true, scale, None)
    }
}
