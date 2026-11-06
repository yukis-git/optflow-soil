//! Diagnostic: `optflow pair` (default config, scale 1) on images with iid Gaussian noise of
//! `eps` x the image RMS added to each pixel, to measure how far an FFT-sized perturbation moves
//! the result (the GPU-vs-CPU baseline of task 13 debug).
//!
//!     cargo run --release -p optflow-cli --example perturb_pair [--features gpu] -- \
//!         REF DEF OUT EPS SEED [cpu|model|perturbed|gpu] [X0 Y0 W H]
//!
//! `EPS` 0 runs the unperturbed images. `SEED` picks the noise (the two images get different
//! streams). The optional rectangle crops both images first. Writes the same maps as the CLI
//! plus `raw_ux.npy`, `raw_uy.npy` (the fine flow before the validity mask).

use optflow_cli::io::{read_image, save_result, write_npy_f32};
use optflow_core::config::Config;
use optflow_core::engine::{CpuEngine, FineEngine};
use optflow_core::gpu_model::{ModelEngine, PerturbedModelEngine};
use optflow_core::image::Image;
use optflow_core::track::track_pair_owned_on;
use std::path::Path;

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn gauss(z: &mut u64) -> f64 {
    let u1 = ((splitmix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((splitmix(z) >> 11) as f64) / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn crop(im: Image<f32>, r: Option<[usize; 4]>) -> Image<f32> {
    let Some([x0, y0, w, h]) = r else { return im };
    let mut d = Vec::with_capacity(w * h);
    for y in y0..y0 + h {
        d.extend_from_slice(&im.data[y * im.w + x0..y * im.w + x0 + w]);
    }
    Image::from_vec(w, h, d)
}

fn perturb(im: &mut Image<f32>, eps: f64, seed: u64) {
    if eps == 0.0 {
        return;
    }
    let n = im.data.len() as f64;
    let rms = (im.data.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n).sqrt();
    let mut z = seed;
    for v in im.data.iter_mut() {
        *v = (f64::from(*v) + eps * rms * gauss(&mut z)) as f32;
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() >= 5,
        "usage: REF DEF OUT EPS SEED [engine] [X0 Y0 W H]"
    );
    let eps: f64 = a[3].parse()?;
    let seed: u64 = a[4].parse()?;
    let which = a.get(5).map(String::as_str).unwrap_or("cpu");
    let rect = if a.len() >= 10 {
        Some([a[6].parse()?, a[7].parse()?, a[8].parse()?, a[9].parse()?])
    } else {
        None
    };
    let engine: Box<dyn FineEngine> = match which {
        "cpu" => Box::new(CpuEngine),
        "model" => Box::new(ModelEngine),
        "perturbed" => Box::new(PerturbedModelEngine(seed)),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        "gpu" => Box::new(optflow_gpu::GpuEngine::new()?),
        e => anyhow::bail!("engine {e}?"),
    };
    let mut i1 = crop(read_image(Path::new(&a[0]))?.into_f32(), rect);
    let mut i2 = crop(read_image(Path::new(&a[1]))?.into_f32(), rect);
    perturb(&mut i1, eps, seed.wrapping_mul(2).wrapping_add(1));
    perturb(&mut i2, eps, seed.wrapping_mul(2).wrapping_add(2));
    let cfg = Config::default();
    let res = track_pair_owned_on(&*engine, i1, i2, &cfg, None, None)?;
    std::fs::create_dir_all(&a[2])?;
    save_result(Path::new(&a[2]), &res)?;
    write_npy_f32(&Path::new(&a[2]).join("raw_ux.npy"), &res.raw_ux)?;
    write_npy_f32(&Path::new(&a[2]).join("raw_uy.npy"), &res.raw_uy)?;
    eprintln!("done {:?}", res.timing.fine);
    Ok(())
}
