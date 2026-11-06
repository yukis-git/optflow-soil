//! The Metal arithmetic equals its Rust mirror bit for bit (this is also the proof that
//! `fp contract(off)` is in effect: a fused a*b+c would change the low bits), and `/` and `sqrt`
//! are correctly rounded.
#![cfg(target_os = "macos")]
use optflow_core::gpu_model::atan2f::atan2f;
use optflow_core::gpu_model::df64::Df64;
use optflow_gpu::device::Gpu;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// uniform in [0, 1)
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// sign * 10^U(-4, 4)
    fn wide(&mut self) -> f32 {
        let m = 10f64.powf(self.unit() * 8.0 - 4.0);
        (if self.next() & 1 == 0 { m } else { -m }) as f32
    }
}

fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn run2(gpu: &Gpu, kernel: &'static str, a: &[f32], b: &[f32], per: usize) -> Vec<f32> {
    let n = a.len();
    let mut ba = gpu.buffer(4 * n).unwrap();
    ba.as_mut_slice::<f32>().copy_from_slice(a);
    let mut bb = gpu.buffer(4 * n).unwrap();
    bb.as_mut_slice::<f32>().copy_from_slice(b);
    let mut out = gpu.buffer(4 * per * n).unwrap();
    gpu.dispatch(
        kernel,
        &mut [&mut ba, &mut bb, &mut out],
        &(n as u32).to_le_bytes(),
        (n, 1, 1),
        (256, 1, 1),
    )
    .unwrap();
    out.as_slice::<f32>().to_vec()
}

#[test]
fn atan2f_gpu_equals_rust_bit_for_bit() {
    let gpu = Gpu::new().unwrap();
    let n = 1_000_000;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut y: Vec<f32> = (0..n).map(|_| rng.wide()).collect();
    let mut x: Vec<f32> = (0..n).map(|_| rng.wide()).collect();
    // special values and axes in the first entries
    let sp = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-30,
        -1e30,
    ];
    for (k, (&a, &b)) in sp
        .iter()
        .flat_map(|a| sp.iter().map(move |b| (a, b)))
        .enumerate()
    {
        y[k] = a;
        x[k] = b;
    }
    let o = run2(&gpu, "test_atan2f", &y, &x, 1);
    let bad: Vec<usize> = (0..n)
        .filter(|&i| !same_bits(o[i], atan2f(y[i], x[i])))
        .collect();
    assert!(
        bad.is_empty(),
        "{} mismatches, first: atan2f({:e}, {:e}) gpu {:e} cpu {:e}",
        bad.len(),
        y[bad[0]],
        x[bad[0]],
        o[bad[0]],
        atan2f(y[bad[0]], x[bad[0]])
    );
}

#[test]
fn df64_gpu_equals_rust_bit_for_bit() {
    let gpu = Gpu::new().unwrap();
    let n = 1_000_000;
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    // values near 1e2..1e3 with an f64 offset so that the lo parts are populated
    let mut input = Vec::with_capacity(5 * n);
    let mut cases = Vec::with_capacity(n);
    for _ in 0..n {
        let a = Df64::from_f64(f64::from(rng.wide()) + rng.unit());
        let b = Df64::from_f64(f64::from(rng.wide()) * 1e-2 + rng.unit());
        let v = rng.wide();
        input.extend_from_slice(&[a.hi, a.lo, b.hi, b.lo, v]);
        cases.push((a, b, v));
    }
    let mut bi = gpu.buffer(4 * 5 * n).unwrap();
    bi.as_mut_slice::<f32>().copy_from_slice(&input);
    let mut out = gpu.buffer(4 * 12 * n).unwrap();
    gpu.dispatch(
        "test_df64",
        &mut [&mut bi, &mut out],
        &(n as u32).to_le_bytes(),
        (n, 1, 1),
        (256, 1, 1),
    )
    .unwrap();
    let o = out.as_slice::<f32>();
    let mut bad = [0usize; 6];
    for (i, &(a, b, v)) in cases.iter().enumerate() {
        let g = &o[12 * i..12 * i + 12];
        let pair = |r: Df64, k: usize| same_bits(g[k], r.hi) && same_bits(g[k + 1], r.lo);
        bad[0] += usize::from(!pair(a.add(b), 0));
        bad[1] += usize::from(!pair(a.sub(b), 2));
        bad[2] += usize::from(!pair(a.mul(b), 4));
        bad[3] += usize::from(!pair(a.mul_f32(v), 6));
        bad[4] += usize::from(!same_bits(g[8], a.diff_f32(b)));
        if a.hi.abs() < 8_388_608.0 {
            let (fl, fr) = a.floor_frac();
            bad[5] += usize::from(!(same_bits(g[9], fr) && g[10] == fl as f32));
        }
    }
    assert_eq!(
        bad, [0; 6],
        "mismatches of add, sub, mul, mul_f32, diff_f32, floor_frac"
    );
}

#[test]
fn division_is_correctly_rounded() {
    let gpu = Gpu::new().unwrap();
    let n = 1_000_000;
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let a: Vec<f32> = (0..n).map(|_| rng.wide().abs()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.wide()).collect();
    let o = run2(&gpu, "test_div_sqrt", &a, &b, 2);
    let bad = (0..n)
        .filter(|&i| o[2 * i].to_bits() != (a[i] / b[i]).to_bits())
        .count();
    println!("division mismatches: {bad}/{n}");
    assert_eq!(bad, 0);
}

/// With `mathFloatingPointFunctions = Precise` (set in `Gpu::new`) `sqrt` is correctly rounded;
/// with the default (fast) setting about 27% of random inputs were off by up to 2 ulp.
#[test]
fn sqrt_is_correctly_rounded() {
    let gpu = Gpu::new().unwrap();
    let n = 1_000_000;
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let a: Vec<f32> = (0..n).map(|_| rng.wide().abs()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.wide()).collect();
    let o = run2(&gpu, "test_div_sqrt", &a, &b, 2);
    let bad = (0..n)
        .filter(|&i| o[2 * i + 1].to_bits() != a[i].sqrt().to_bits())
        .count();
    println!("sqrt mismatches: {bad}/{n}");
    assert_eq!(bad, 0);
}
