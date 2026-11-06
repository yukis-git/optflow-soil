#![cfg(target_os = "macos")]
use optflow_gpu::device::Gpu;

#[test]
fn safe_math_keeps_two_sum_and_nan() {
    let gpu = Gpu::new().expect("Metal device");
    let n = 4096;
    let a: Vec<f32> = (0..n).map(|i| 1000.0 + i as f32 * 0.37).collect();
    let b: Vec<f32> = (0..n).map(|i| 1e-4 * (i as f32 * 0.11).sin()).collect();
    let mut ba = gpu.buffer(4 * n).unwrap();
    ba.as_mut_slice::<f32>().copy_from_slice(&a);
    let mut bb = gpu.buffer(4 * n).unwrap();
    bb.as_mut_slice::<f32>().copy_from_slice(&b);
    bb.as_mut_slice::<f32>()[7] = f32::NAN;
    let mut out = gpu.buffer(4 * 3 * n).unwrap();
    gpu.dispatch(
        "test_two_sum",
        &mut [&mut ba, &mut bb, &mut out],
        &(n as u32).to_le_bytes(),
        (n, 1, 1),
        (256, 1, 1),
    )
    .unwrap();
    let o = out.as_slice::<f32>();
    for i in 0..n {
        let (s, e, isnan) = (o[3 * i], o[3 * i + 1], o[3 * i + 2]);
        if i == 7 {
            assert_eq!(isnan, 1.0);
            continue;
        }
        assert_eq!(isnan, 0.0);
        assert_eq!(
            f64::from(s) + f64::from(e),
            f64::from(a[i]) + f64::from(b[i]),
            "two_sum at {i}"
        );
    }
}

#[test]
fn a_failing_command_is_an_error_not_a_panic() {
    let gpu = Gpu::new().unwrap();
    let r = gpu.dispatch("no_such_kernel", &mut [], &[], (1, 1, 1), (1, 1, 1));
    assert!(r.is_err());
}

#[test]
fn pooled_buffers_come_back_zeroed() {
    let gpu = Gpu::new().unwrap();
    let mut b = gpu.buffer(64).unwrap();
    b.as_mut_slice::<u8>().fill(0xAB);
    gpu.release(b);
    let b2 = gpu.buffer(64).unwrap();
    assert!(b2.as_slice::<u8>().iter().all(|&x| x == 0));
}

#[test]
fn device_name_and_limits() {
    assert!(!optflow_gpu::device_name().unwrap().is_empty());
    let gpu = Gpu::new().unwrap();
    assert!(gpu.max_buffer_length() > 0);
    assert!(gpu.recommended_working_set() > 0);
}

#[test]
fn oversized_params_and_allocations_are_errors() {
    let gpu = Gpu::new().unwrap();
    let r = gpu.dispatch("test_two_sum", &mut [], &[0u8; 5000], (1, 1, 1), (1, 1, 1));
    assert!(r.is_err());
    assert!(gpu.buffer(usize::MAX / 2).is_err());
}
