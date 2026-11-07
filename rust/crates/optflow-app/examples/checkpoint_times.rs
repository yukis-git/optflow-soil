//! Reads a checkpoint and times it, and writes the same volume as raw arrays for comparison: the
//! baseline of the notes, every array of the state at full size with no saving.
//!
//! `checkpoint_times FILE WIDTH HEIGHT BYTES_PER_PIXEL`

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use optflow_cli::checkpoint::{summary, write_durably};
use optflow_core::checkpoint::{Data, Reader};
use optflow_core::track::resume::SavedRun;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = Path::new(&args[0]);
    let (w, h): (u64, u64) = (args[1].parse().unwrap(), args[2].parse().unwrap());
    let per_px: f64 = args[3].parse().unwrap();
    let s = summary(path).unwrap();
    println!("file {} bytes, frame {}", s.bytes, s.head.k);
    for _ in 0..3 {
        let t = Instant::now();
        let mut r = Reader::open(File::open(path).unwrap()).unwrap();
        let state = SavedRun::read(&mut r).unwrap();
        println!("read and check {:.3} s", t.elapsed().as_secs_f64());
        drop(state);
    }
    let n = (w as f64 * h as f64 * per_px) as usize;
    let data = vec![0x5Au8; n];
    let (tmp, dest) = (path.with_extension("base.tmp"), path.with_extension("base"));
    for _ in 0..3 {
        let t = Instant::now();
        write_durably(&tmp, &dest, &[(1, Data::Bytes(&data))]).unwrap();
        println!(
            "baseline {n} bytes: write {:.3} s",
            t.elapsed().as_secs_f64()
        );
        let t = Instant::now();
        let mut r = Reader::open(File::open(&dest).unwrap()).unwrap();
        let back = r.bytes(1).unwrap();
        println!("baseline read and check {:.3} s", t.elapsed().as_secs_f64());
        assert_eq!(back.len(), n);
    }
    let _ = std::fs::remove_file(&dest);
}
