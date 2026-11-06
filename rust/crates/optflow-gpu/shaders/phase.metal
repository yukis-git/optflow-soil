// Reference phase, bank responses and the anchor-free accumulate: ports of
// optflow-core/src/{gabor.rs, phaseflow.rs} (padding, filter products, RefPack, Responses layout)
// and gpu_model/{field.rs, solve.rs} (sample_centred, pixel_sums), same operation order.

constant uint PHASE_PAD = 32;

// ---- padding, filter products, scaling ----

struct PadP { uint w; uint h; uint pw; uint ph; };

// gabor::pad_reflect as complex values with zero imaginary part
kernel void pad_reflect_complex(device const float* img [[buffer(0)]], device float* out [[buffer(1)]],
                                constant PadP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.pw || g.y >= p.ph) return;
    int sx = mirror_idx((int)g.x - (int)PHASE_PAD, (int)p.w);
    int sy = mirror_idx((int)g.y - (int)PHASE_PAD, (int)p.h);
    uint o = g.y * p.pw + g.x;
    out[2 * o] = img[sy * p.w + sx];
    out[2 * o + 1] = 0.0f;
}

struct MulP { uint n; uint count; uint pw; uint ph; uint axis; uint pad0; uint pad1; uint pad2; };

// out[k][i] = spectrum[i] * filters[k][i] (complex times real)
kernel void filter_multiply(device const float* spec [[buffer(0)]], device const float* filt [[buffer(1)]],
                            device float* out [[buffer(2)]], constant MulP& p [[buffer(3)]],
                            uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.n || g.y >= p.count) return;
    float f = filt[g.y * p.n + g.x];
    uint o = g.y * p.n + g.x;
    out[2 * o] = spec[2 * g.x] * f;
    out[2 * o + 1] = spec[2 * g.x + 1] * f;
}

// the product above times i * f along the axis (gabor::reference_inverses' derivative); freqs is
// fx (pw values) then fy (ph values)
kernel void filter_multiply_deriv(device const float* spec [[buffer(0)]], device const float* filt [[buffer(1)]],
                                  device const float* freqs [[buffer(2)]], device float* out [[buffer(3)]],
                                  constant MulP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.n || g.y >= p.count) return;
    uint o = g.y * p.n + g.x;
    float f = filt[o];
    float re = spec[2 * g.x] * f;
    float im = spec[2 * g.x + 1] * f;
    float fr = (p.axis == 0) ? freqs[g.x % p.pw] : freqs[p.pw + g.x / p.pw];
    out[2 * o] = (-im) * fr;
    out[2 * o + 1] = re * fr;
}

struct ScaleP { uint n; float s; };

kernel void scale_inplace(device float* data [[buffer(0)]], constant ScaleP& p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    data[i] *= p.s;
}

// ---- reference_values in Df64, rounded once ----

inline df64 df_prod(float a, float b) {
    float2 p = two_prod(a, b);
    return df64{p.x, p.y};
}

// n / d rounded to f32 (n, d Df64): the quotient of the leading parts plus the quotient of the
// remainder
inline float df_div_round(df64 n, df64 d) {
    float q = n.hi / d.hi;
    df64 rem = df_sub(n, df_mul_f(d, q));
    return q + rem.hi / d.hi;
}

inline float df_sqrt_round(df64 a) {
    float s = sqrt(a.hi);
    df64 e = df_sub(a, df_prod(s, s));
    return s + e.hi / (2.0f * s);
}

inline float3 reference_values(float2 r, float2 rx, float2 ry, df64 eps) {
    df64 a2 = df_add(df_add(df_prod(r.x, r.x), df_prod(r.y, r.y)), eps);
    df64 nx = df_sub(df_prod(r.x, rx.y), df_prod(r.y, rx.x));
    df64 ny = df_sub(df_prod(r.x, ry.y), df_prod(r.y, ry.x));
    return float3(df_div_round(nx, a2), df_div_round(ny, a2), df_sqrt_round(a2));
}

struct RefP { uint w; uint h; uint pw; uint ph; uint count; uint first; uint nf; uint pad0;
              float eps_hi; float eps_lo; };

// thread = pixel; for each of the `count` filters the entry [r.re, r.im, gx, gy, amp]
kernel void reference_pack(device const float* r [[buffer(0)]], device const float* rx [[buffer(1)]],
                           device const float* ry [[buffer(2)]], device float* out [[buffer(3)]],
                           constant RefP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    uint i = g.y * p.w + g.x;
    uint q = (g.y + PHASE_PAD) * p.pw + g.x + PHASE_PAD;
    uint plane = p.pw * p.ph;
    df64 eps = df64{p.eps_hi, p.eps_lo};
    for (uint j = 0; j < p.count; j++) {
        uint s = 2 * (j * plane + q);
        float2 rv = float2(r[s], r[s + 1]);
        float3 v = reference_values(rv, float2(rx[s], rx[s + 1]), float2(ry[s], ry[s + 1]), eps);
        device float* o = out + (i * p.nf + p.first + j) * 5;
        o[0] = rv.x; o[1] = rv.y; o[2] = v.x; o[3] = v.y; o[4] = v.z;
    }
}

struct InterP { uint n; uint count; uint first; uint nf; };

// planes (count complex planes of n nodes) -> node layout out[(node * nf + first + j)]
kernel void interleave_filters(device const float* planes [[buffer(0)]], device float* out [[buffer(1)]],
                               constant InterP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.n || g.y >= p.count) return;
    uint s = 2 * (g.y * p.n + g.x);
    uint o = 2 * (g.x * p.nf + p.first + g.y);
    out[o] = planes[s];
    out[o + 1] = planes[s + 1];
}

// ---- accumulate ----

// field::weights_f32
inline void weights_f32(float t, thread float* w) {
    float u = 1.0f - t;
    float t2 = t * t;
    float t3 = t2 * t;
    w[0] = u * u * u / 6.0f;
    w[1] = (3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f;
    w[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
    w[3] = t3 / 6.0f;
}

// field::axis_taps_df64 (requires a finite coordinate)
inline void axis_taps(df64 c, uint n, thread int* idx, thread float* w) {
    int ci;
    float t = df_floor_frac(c, ci);
    float tt;
    int c0 = clamp_axis(ci, t, n, tt);
    for (int k = 0; k < 4; k++) idx[k] = mirror_idx(c0 + k - 1, (int)n);
    weights_f32(tt, w);
}

// field::sample_centred
inline df64 sample_centred(device const float* c, uint w, uint h, df64 x, df64 y) {
    if (!(isfinite(x.hi) && isfinite(y.hi))) return df64{NAN, NAN};
    int xi[4]; int yi[4]; float wx[4]; float wy[4];
    axis_taps(x, w, xi, wx);
    axis_taps(y, h, yi, wy);
    float base = c[yi[0] * w + xi[0]];
    float total = 0.0f;
    for (int j = 0; j < 4; j++) {
        device const float* row = c + yi[j] * w;
        float acc = wx[0] * (row[xi[0]] - base);
        for (int k = 1; k < 4; k++) acc += wx[k] * (row[xi[k]] - base);
        total += wy[j] * acc;
    }
    return df_add_f(df_from_f32(base), total);
}

// one filter's terms of solve::pixel_sums, added to s[0..7] (gxx, gxy, gyy, hx, hy, s_phiphi, s_w)
inline void add_filter(float b_re, float b_im, float ar, float ai, float gx, float gy, float amp,
                       thread float* s) {
    float num = ai * b_re - ar * b_im;
    float den = ar * b_re + ai * b_im;
    float mag = sqrt(b_re * b_re + b_im * b_im);
    float dphi = atan2f(num, den);
    float wk = amp * mag;
    s[0] += (wk * gx) * gx;
    s[1] += (wk * gx) * gy;
    s[2] += (wk * gy) * gy;
    s[3] += (wk * gx) * dphi;
    s[4] += (wk * gy) * dphi;
    s[5] += (wk * dphi) * dphi;
    s[6] += wk;
}

struct AccP { uint w; uint h; uint pw; uint ph; uint nf; uint pad0; uint pad1; uint pad2; };

// solve::pixel_sums; sums = 7 planes of w*h: gxx, gxy, gyy, hx, hy, s_phiphi, s_w
kernel void accumulate(device const float* refpack [[buffer(0)]], device const float* resp [[buffer(1)]],
                       device const float* sx [[buffer(2)]], device const float* sy [[buffer(3)]],
                       device const float* csx [[buffer(4)]], device const float* csy [[buffer(5)]],
                       device const float* ux_hi [[buffer(6)]], device const float* ux_lo [[buffer(7)]],
                       device const float* uy_hi [[buffer(8)]], device const float* uy_lo [[buffer(9)]],
                       device float* sums [[buffer(10)]], constant AccP& p [[buffer(11)]],
                       uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    uint i = g.y * p.w + g.x;
    uint n = p.w * p.h;
    df64 tx = df_add_f(df64{ux_hi[i], ux_lo[i]}, (float)g.x);
    df64 ty = df_add_f(df64{uy_hi[i], uy_lo[i]}, (float)g.y);
    df64 qx = df_add_f(tx, -sx[i]);
    df64 qy = df_add_f(ty, -sy[i]);
    for (int it = 0; it < 2; it++) {
        df64 dx = sample_centred(csx, p.w, p.h, qx, qy);
        df64 dy = sample_centred(csy, p.w, p.h, qx, qy);
        qx = df_sub(tx, dx);
        qy = df_sub(ty, dy);
    }
    df64 px = df_add_f(qx, (float)PHASE_PAD);
    df64 py = df_add_f(qy, (float)PHASE_PAD);
    if (!(isfinite(px.hi) && isfinite(py.hi))) {
        for (uint s = 0; s < 7; s++) sums[s * n + i] = NAN;
        return;
    }
    int xi[4]; int yi[4]; float wx[4]; float wy[4];
    axis_taps(px, p.pw, xi, wx);
    axis_taps(py, p.ph, yi, wy);
    uint nf = p.nf;
    device const float* rp = refpack + i * nf * 5;
    float s[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < nf; k++) {
        float b_re = 0.0f, b_im = 0.0f;
        for (int j = 0; j < 4; j++) {
            device const float* line = resp + (uint)yi[j] * p.pw * nf * 2;
            device const float* c0 = line + ((uint)xi[0] * nf + k) * 2;
            float a_re = wx[0] * c0[0];
            float a_im = wx[0] * c0[1];
            for (int t = 1; t < 4; t++) {
                device const float* c = line + ((uint)xi[t] * nf + k) * 2;
                a_re += wx[t] * c[0];
                a_im += wx[t] * c[1];
            }
            b_re += wy[j] * a_re;
            b_im += wy[j] * a_im;
        }
        device const float* e = rp + 5 * k;
        add_filter(b_re, b_im, e[0], e[1], e[2], e[3], e[4], s);
    }
    sums[i] = s[0]; sums[n + i] = s[1]; sums[2 * n + i] = s[2]; sums[3 * n + i] = s[3];
    sums[4 * n + i] = s[4]; sums[5 * n + i] = s[5]; sums[6 * n + i] = s[6];
}
