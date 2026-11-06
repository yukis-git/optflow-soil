//! The Df64 prefilter (checkpointed causal pass, segmented anti-causal pass) against a plain
//! host recursion that stores every causal value, bit for bit; and the block-count limit of
//! the local kernels.
#![cfg(target_os = "macos")]
use optflow_core::gpu_model::df64::Df64;
use optflow_gpu::device::{Buffer, Gpu};
use optflow_gpu::kernels::{self, MomentPlanGpu};

const HORIZON: usize = 21;

fn upload(gpu: &Gpu, v: &[f32]) -> Buffer {
    let mut b = gpu.buffer(4 * v.len()).unwrap();
    b.as_mut_slice::<f32>().copy_from_slice(v);
    b
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2000.0
        })
        .collect()
}

/// `interp::prefilter_line` in Df64 as before the checkpointing: every causal value is stored.
fn line(x: &[Df64]) -> Vec<Df64> {
    let n = x.len();
    if n == 1 {
        return x.to_vec();
    }
    let z = 3.0f64.sqrt() - 2.0;
    let scale = Df64::from_f64(z / (z * z - 1.0));
    let zd = Df64::from_f64(z);
    let six = Df64::from_f32(6.0);
    let mut zk = 1.0f64;
    let mut c = vec![Df64::from_f32(0.0); n];
    let mut sum = x[0].mul(six);
    for xi in x.iter().take(n.min(HORIZON)).skip(1) {
        zk *= z;
        sum = sum.add(Df64::from_f64(zk).mul(xi.mul(six)));
    }
    c[0] = sum;
    for i in 1..n {
        c[i] = x[i].mul(six).add(zd.mul(c[i - 1]));
    }
    c[n - 1] = scale.mul(c[n - 1].add(zd.mul(c[n - 2])));
    for i in (0..n - 1).rev() {
        c[i] = zd.mul(c[i + 1].sub(c[i]));
    }
    c
}

/// Host prefilter of `planes` planes of `w x h` elements of `es` floats.
fn reference(src: &[f32], (w, h, es, planes): (usize, usize, usize, usize)) -> Vec<f32> {
    let plane = w * h * es;
    let mut out = vec![0.0f32; src.len()];
    for p in 0..planes {
        for lane in 0..es {
            let at = |x: usize, y: usize| p * plane + (y * w + x) * es + lane;
            let mut t = vec![Df64::from_f32(0.0); w * h];
            for x in 0..w {
                let col: Vec<_> = (0..h).map(|y| Df64::from_f32(src[at(x, y)])).collect();
                for (y, v) in line(&col).into_iter().enumerate() {
                    t[y * w + x] = v;
                }
            }
            for y in 0..h {
                for (x, v) in line(&t[y * w..(y + 1) * w]).into_iter().enumerate() {
                    out[at(x, y)] = v.hi + v.lo;
                }
            }
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Lengths congruent to 0, 1 and 15 modulo the checkpoint spacing 16, and short ones.
#[test]
fn df64_prefilter_matches_the_plain_recursion() {
    let gpu = Gpu::new().unwrap();
    for &(w, h) in &[
        (16usize, 17usize),
        (17, 16),
        (31, 47),
        (47, 31),
        (64, 65),
        (65, 64),
        (1, 31),
        (15, 1),
        (2, 33),
    ] {
        let n = w * h;
        let count = 2;
        let data = values(2 * n * count, (w * 131 + h) as u64);
        let mut a = upload(&gpu, &data);
        let mut scratch = gpu.buffer(16 * n * count).unwrap();
        gpu.batch(|b| kernels::prefilter_complex(b, &mut a, &mut scratch, w, h, count))
            .unwrap()
            .unwrap();
        let want = reference(&data, (w, h, 2, count));
        assert_eq!(bits(a.as_slice::<f32>()), bits(&want), "complex {w}x{h}");

        let img = values(n, (w * 7 + h) as u64);
        let mut src = upload(&gpu, &img);
        let mut scratch = gpu.buffer(8 * n).unwrap();
        let mut out = gpu.buffer(4 * n).unwrap();
        gpu.batch(|b| kernels::prefilter_real(b, &mut src, &mut scratch, &mut out, w, h))
            .unwrap()
            .unwrap();
        let want = reference(&img, (w, h, 1, 1));
        assert_eq!(bits(out.as_slice::<f32>()), bits(&want), "real {w}x{h}");
    }
}

/// More blocks than the z extent of the local kernels' grid is an error, not a bad dispatch.
#[test]
fn solve_blocks_rejects_too_many_blocks() {
    let gpu = Gpu::new().unwrap();
    let (w, h) = (16 * 65_536, 1);
    let (mut taps, r) = kernels::moment_taps_buffer(&gpu, 5.5).unwrap();
    let mut plan = MomentPlanGpu::new(&gpu, 1).unwrap();
    let tiny = || gpu.buffer(16).unwrap();
    let (mut sums, mut global, mut slope, mut erange) = (tiny(), tiny(), tiny(), tiny());
    let mut ui = [0; 4].map(|_| tiny());
    let mut uo = [0; 4].map(|_| tiny());
    let (mut energy, mut resid) = (tiny(), tiny());
    let [i0, i1, i2, i3] = &mut ui;
    let [o0, o1, o2, o3] = &mut uo;
    let err = gpu
        .batch(|b| {
            kernels::solve_blocks(
                b,
                &mut sums,
                &mut global,
                [i0, i1, i2, i3],
                w,
                h,
                &mut taps,
                r,
                &mut plan,
                [o0, o1, o2, o3],
                &mut energy,
                &mut resid,
                &mut slope,
                &mut erange,
            )
        })
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("blocks exceed"), "{err}");
}
