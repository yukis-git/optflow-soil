#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

// Double-float helpers and atan2f: line-by-line ports of
// optflow-core/src/gpu_model/{df64,atan2f}.rs. Keep the same operation order. `fma` is the only
// fused operation (the Rust mirror uses explicit mul_add); contraction of a*b+c is switched off
// above.

struct df64 { float hi; float lo; };

inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
inline float2 two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}
inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

inline df64 df_from_f32(float v) { return df64{v, 0.0f}; }

inline df64 df_add(df64 x, df64 y) {
    float2 se = two_sum(x.hi, y.hi);
    float2 tf = two_sum(x.lo, y.lo);
    float2 s = quick_two_sum(se.x, se.y + tf.x);
    float2 r = quick_two_sum(s.x, s.y + tf.y);
    return df64{r.x, r.y};
}
inline df64 df_sub(df64 x, df64 y) { return df_add(x, df64{-y.hi, -y.lo}); }
inline df64 df_add_f(df64 x, float v) { return df_add(x, df_from_f32(v)); }
inline df64 df_mul(df64 x, df64 y) {
    float2 pe = two_prod(x.hi, y.hi);
    float2 r = quick_two_sum(pe.x, pe.y + (x.hi * y.lo + x.lo * y.hi));
    return df64{r.x, r.y};
}
inline df64 df_mul_f(df64 x, float v) {
    float2 pe = two_prod(x.hi, v);
    float2 r = quick_two_sum(pe.x, fma(x.lo, v, pe.y));
    return df64{r.x, r.y};
}
// (x - y) rounded to f32: for differences of nearby values.
inline float df_diff_f(df64 x, df64 y) {
    df64 d = df_sub(x, y);
    return d.hi + d.lo;
}
// Integer part (floor, saturated like Rust's `as i32`) and fraction in [0, 1); requires |value| < 2^23.
inline float df_floor_frac(df64 x, thread int& ipart) {
    float fl = floor(x.hi);
    float f = (x.hi - fl) + x.lo;
    if (f < 0.0f) {
        fl -= 1.0f;
        f += 1.0f;
    }
    if (f >= 1.0f) {
        f = as_type<float>(as_type<uint>(1.0f) - 1u);
    }
    ipart = int(clamp(fl, -2147483648.0f, 2147483520.0f));
    return f;
}

constant float ATAN_SQRT3_HI = 1.7320508f;
constant float ATAN_SQRT3_LO = 3.108725e-8f;
constant float ATAN_T_REDUCE = 0.2679492f;
constant float ATAN_PIO6_HI = 0.5235988f;
constant float ATAN_PIO6_LO = -1.4570463e-8f;
constant float ATAN_PIO2_HI = 1.5707964f;
constant float ATAN_PIO2_LO = -4.371139e-8f;
constant float ATAN_PI_HI = 3.1415927f;
constant float ATAN_PI_LO = -8.742278e-8f;
constant float ATAN_P0 = -0.33333333f;
constant float ATAN_P1 = 0.1999994f;
constant float ATAN_P2 = -0.14281865f;
constant float ATAN_P3 = 0.1100113f;
constant float ATAN_P4 = -0.07637197f;

inline float atan2f(float y, float x) {
    if (isnan(y) || isnan(x)) {
        return NAN;
    }
    float ax = fabs(x);
    float ay = fabs(y);
    if (ax == INFINITY || ay == INFINITY) {
        ax = (ax == INFINITY) ? 1.0f : 0.0f;
        ay = (ay == INFINITY) ? 1.0f : 0.0f;
    }
    bool swp = ay > ax;
    float n = swp ? ax : ay;
    float d = swp ? ay : ax;
    float t = (d == 0.0f) ? 0.0f : n / d;
    bool reduce = t > ATAN_T_REDUCE;
    float u;
    if (reduce) {
        float num = fma(ATAN_SQRT3_HI, n, -d) + ATAN_SQRT3_LO * n;
        float den = fma(ATAN_SQRT3_HI, d, n) + ATAN_SQRT3_LO * d;
        u = num / den;
    } else {
        u = t;
    }
    float z = u * u;
    float p = fma(fma(fma(fma(ATAN_P4, z, ATAN_P3), z, ATAN_P2), z, ATAN_P1), z, ATAN_P0);
    float s;
    float c0;
    if (reduce) {
        float2 se = two_sum(ATAN_PIO6_HI, u);
        s = se.x;
        c0 = se.y + ATAN_PIO6_LO;
    } else {
        s = u;
        c0 = 0.0f;
    }
    float corr = c0 + (u * z) * p;
    bool left = signbit(x);
    float bh;
    float bl;
    if (swp) {
        bh = ATAN_PIO2_HI;
        bl = ATAN_PIO2_LO;
    } else if (left) {
        bh = ATAN_PI_HI;
        bl = ATAN_PI_LO;
    } else {
        bh = 0.0f;
        bl = 0.0f;
    }
    float ss;
    float sc;
    if (swp == left) {
        ss = s;
        sc = corr;
    } else {
        ss = -s;
        sc = -corr;
    }
    float2 he = two_sum(bh, ss);
    float res = he.x + ((he.y + bl) + sc);
    return copysign(res, y);
}

// ---- test kernels (cheap; used by tests/device.rs and tests/arith.rs) ----

kernel void test_two_sum(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float2 s = two_sum(a[i], b[i]);
    out[3 * i] = s.x; out[3 * i + 1] = s.y; out[3 * i + 2] = isnan(b[i]) ? 1.0f : 0.0f;
}

kernel void test_atan2f(device const float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = atan2f(y[i], x[i]);
}

// in: 5 floats per case (a.hi, a.lo, b.hi, b.lo, v); out: 12 floats per case:
// add(2), sub(2), mul(2), mul_f(2), diff_f(1), floor_frac(frac, ipart-as-float)(2), spare(1)
kernel void test_df64(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    df64 a = df64{in[5 * i], in[5 * i + 1]};
    df64 b = df64{in[5 * i + 2], in[5 * i + 3]};
    float v = in[5 * i + 4];
    device float* o = out + 12 * i;
    df64 r = df_add(a, b);
    o[0] = r.hi; o[1] = r.lo;
    r = df_sub(a, b);
    o[2] = r.hi; o[3] = r.lo;
    r = df_mul(a, b);
    o[4] = r.hi; o[5] = r.lo;
    r = df_mul_f(a, v);
    o[6] = r.hi; o[7] = r.lo;
    o[8] = df_diff_f(a, b);
    int ip;
    o[9] = df_floor_frac(a, ip);
    o[10] = float(ip);
    o[11] = 0.0f;
}

// out: a / b, sqrt(a)
kernel void test_div_sqrt(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[2 * i] = a[i] / b[i];
    out[2 * i + 1] = sqrt(a[i]);
}
