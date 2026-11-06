
// B-spline prefilters, centred field blur and the Df64 warp: ports of
// optflow-core/src/interp.rs (prefilter_line) and gpu_model/field.rs, same operation order.

// ---- recursive prefilter (one thread per line) ----

struct PrefilterP { uint w; uint h; uint es; uint planes; float z; float scale; };

constant uint HORIZON = 21;
// elements per load batch of the Df64 prefilter passes
constant uint PF = 8;
// segment of the anti-causal pass (every PK-th causal value is stored)
constant uint PK = 16;

// interp::prefilter_line in f32 on the strided line c[0], c[s], ...
inline void prefilter_line(device float* c, uint s, uint n, float z, float scale) {
    if (n == 1) return;
    for (uint i = 0; i < n; i++) c[i * s] *= 6.0f;
    uint h = min(n, HORIZON);
    float zk = 1.0f;
    float sum = c[0];
    for (uint i = 1; i < h; i++) {
        zk *= z;
        sum += zk * c[i * s];
    }
    c[0] = sum;
    for (uint i = 1; i < n; i++) c[i * s] += z * c[(i - 1) * s];
    c[(n - 1) * s] = scale * (c[(n - 1) * s] + z * c[(n - 2) * s]);
    for (uint i = n - 1; i-- > 0;) c[i * s] = z * (c[(i + 1) * s] - c[i * s]);
}

// ---- Df64 prefilter of image planes (the f64 recursion of BSpline::new, rounded once) ----

struct DfPrefilterP {
    uint w; uint h; uint es; uint planes;
    float z_hi; float z_lo; float scale_hi; float scale_lo;
    float zk_hi[HORIZON]; float zk_lo[HORIZON];   // z^k (f64 powers) as Df64
    float mul;      // the y pass reads src * mul (an f32 product: the inverse FFT's scaling)
    uint first;     // prefilter_df_x_nodes: first filter of the planes, filters per node
    uint nf;
    uint pad0;
};

// interp::prefilter_line in Df64 on a strided line, the recursion state in registers. The
// input element i is (in_hi[i s] * mul, in_lo[i s]) (in_lo null: 0); the result goes to (hi, lo)
// or, rounded to f32, to out[i os] when `out` is not null. The operations and their order are
// those of prefilter_line: 6 x, the start sum over HORIZON terms, the causal and anti-causal
// passes. To stream each array as few times as possible, the causal pass stores only every
// PK-th causal value (and the first) in (hi, lo); the anti-causal pass goes back segment by
// segment of PK elements, recomputing the segment's causal values from the stored one before it
// (the same operations, so the same values). (hi, lo) may be the input (in place).
inline df64 pf_in(device const float* in_hi, device const float* in_lo, float mul, uint i, uint s) {
    return df64{in_hi[i * s] * mul, in_lo ? in_lo[i * s] : 0.0f};
}

inline void prefilter_line_df(device const float* in_hi, device const float* in_lo, float mul,
                              device float* hi, device float* lo, device float* out, uint s,
                              uint os, uint n, constant DfPrefilterP& p) {
    if (n == 1) {
        if (out) out[0] = in_hi[0] * mul + (in_lo ? in_lo[0] : 0.0f);
        else { hi[0] = in_hi[0] * mul; lo[0] = in_lo ? in_lo[0] : 0.0f; }
        return;
    }
    df64 z = df64{p.z_hi, p.z_lo};
    df64 scale = df64{p.scale_hi, p.scale_lo};
    df64 six = df_from_f32(6.0f);
    uint h = min(n, HORIZON);
    df64 sum = df_mul(pf_in(in_hi, in_lo, mul, 0, s), six);
    for (uint i = 1; i < h; i++) {
        df64 zk = df64{p.zk_hi[i], p.zk_lo[i]};
        sum = df_add(sum, df_mul(zk, df_mul(pf_in(in_hi, in_lo, mul, i, s), six)));
    }
    df64 prev = sum, pprev = sum;
    // causal; the loads of PF elements are issued together, ahead of the recursion
    for (uint i0 = 1; i0 < n; i0 += PF) {
        uint cnt = min(PF, n - i0);
        df64 x[PF];
        for (uint k = 0; k < PF; k++) x[k] = pf_in(in_hi, in_lo, mul, i0 + min(k, cnt - 1), s);
        for (uint k = 0; k < PF; k++) {
            if (k < cnt) {
                uint i = i0 + k;
                df64 v = df_add(df_mul(x[k], six), df_mul(z, prev));
                if ((i + 1) % PK == 0) { hi[i * s] = v.hi; lo[i * s] = v.lo; }
                pprev = prev;
                prev = v;
            }
        }
    }
    // (after the loop: the input at 0 is no longer read)
    hi[0] = sum.hi; lo[0] = sum.lo;
    df64 next = df_mul(scale, df_add(prev, df_mul(z, pprev)));
    if (out) out[(n - 1) * os] = next.hi + next.lo;
    else { hi[(n - 1) * s] = next.hi; lo[(n - 1) * s] = next.lo; }
    // anti-causal, elements n - 2 .. 0, by segments [j PK, (j + 1) PK)
    uint last = n - 2;
    for (uint j = last / PK + 1; j-- > 0;) {
        uint i0 = j * PK;
        df64 c[PK];
        df64 pv = i0 > 0 ? df64{hi[(i0 - 1) * s], lo[(i0 - 1) * s]} : df64{0.0f, 0.0f};
        for (uint k = 0; k < PK; k++) {
            uint i = min(i0 + k, last);
            df64 x = pf_in(in_hi, in_lo, mul, i, s);
            if (i == 0 || (i + 1) % PK == 0) c[k] = df64{hi[i * s], lo[i * s]};
            else c[k] = df_add(df_mul(x, six), df_mul(z, pv));
            pv = c[k];
        }
        for (uint k = PK; k-- > 0;) {
            uint i = i0 + k;
            if (i > last) continue;
            df64 v = df_mul(z, df_sub(next, c[k]));
            if (out) out[i * os] = v.hi + v.lo;
            else { hi[i * s] = v.hi; lo[i * s] = v.lo; }
            next = v;
        }
    }
}

// pass along y: src (f32 planes of es-float elements) -> scratch (hi plane, then lo plane, same
// layout); thread = (x * es + lane, plane)
kernel void prefilter_df_y(device const float* src [[buffer(0)]], device float* scratch [[buffer(1)]],
                           constant DfPrefilterP& p [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w * p.es || g.y >= p.planes) return;
    uint total = p.planes * p.w * p.h * p.es;
    uint base = g.y * p.w * p.h * p.es + g.x;
    uint s = p.w * p.es;
    prefilter_line_df(src + base, nullptr, p.mul, scratch + base, scratch + total + base, nullptr,
                      s, s, p.h, p);
}

// pass along x in the scratch, then round once to f32 into dst; thread = (y * es + lane, plane)
kernel void prefilter_df_x(device float* scratch [[buffer(0)]], device float* dst [[buffer(1)]],
                           constant DfPrefilterP& p [[buffer(2)]],
                           uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.h * p.es || g.y >= p.planes) return;
    uint total = p.planes * p.w * p.h * p.es;
    uint y = g.x / p.es;
    uint lane = g.x % p.es;
    uint base = g.y * p.w * p.h * p.es + y * p.w * p.es + lane;
    prefilter_line_df(scratch + base, scratch + total + base, 1.0f, scratch + base,
                      scratch + total + base, dst + base, p.es, p.es, p.w, p);
}

// as prefilter_df_x for complex planes (es = 2) of the filters first.. of a node layout: the
// result goes to out[((y * w + x) * nf + first + plane) * 2 + lane] (interleave_filters);
// thread t = (lane, plane, row) with the lane fastest, so that neighbouring threads write
// neighbouring values
kernel void prefilter_df_x_nodes(device float* scratch [[buffer(0)]], device float* out [[buffer(1)]],
                                 constant DfPrefilterP& p [[buffer(2)]],
                                 uint t [[thread_position_in_grid]]) {
    uint lane = t % 2, plane = (t / 2) % p.planes, y = t / (2 * p.planes);
    if (y >= p.h) return;
    uint total = p.planes * p.w * p.h * 2;
    uint base = plane * p.w * p.h * 2 + y * p.w * 2 + lane;
    device float* o = out + (y * p.w * p.nf + p.first + plane) * 2 + lane;
    prefilter_line_df(scratch + base, scratch + total + base, 1.0f, scratch + base,
                      scratch + total + base, o, 2, 2 * p.nf, p.w, p);
}

// field::spline_coeffs_centred, pass along y: out = column prefilter of (s - s[row 0])
kernel void field_prefilter_y(device const float* s [[buffer(0)]], device float* out [[buffer(1)]],
                              constant PrefilterP& p [[buffer(2)]],
                              uint x [[thread_position_in_grid]]) {
    if (x >= p.w) return;
    float base = s[x];
    for (uint y = 0; y < p.h; y++) out[y * p.w + x] = s[y * p.w + x] - base;
    prefilter_line(out + x, p.w, p.h, p.z, p.scale);
}

// pass along x, in place on `out`, re-based per row in Df64
kernel void field_prefilter_x(device const float* s [[buffer(0)]], device float* out [[buffer(1)]],
                              constant PrefilterP& p [[buffer(2)]],
                              uint y [[thread_position_in_grid]]) {
    if (y >= p.h) return;
    device float* row = out + y * p.w;
    df64 row_base = df_add_f(df_from_f32(s[0]), row[0]);
    for (uint x = 0; x < p.w; x++) {
        df64 v = df_add_f(df_from_f32(s[x]), row[x]);
        row[x] = df_diff_f(v, row_base);
    }
    prefilter_line(row, 1, p.w, p.z, p.scale);
    for (uint x = 0; x < p.w; x++) {
        df64 c = df_add_f(row_base, row[x]);
        row[x] = c.hi + c.lo;
    }
}

// ---- centred blur: the taps are a device buffer ----

struct BlurP { uint w; uint h; uint taps; uint pad; };

kernel void blur_y(device const float* u_hi [[buffer(0)]], device const float* u_lo [[buffer(1)]],
                   device const float* k [[buffer(2)]], device float* mid [[buffer(3)]],
                   constant BlurP& p [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    uint r = p.taps / 2;
    uint n = p.w * p.h;
    df64 base = df64{u_hi[g.y * p.w + g.x], u_lo[g.y * p.w + g.x]};
    float acc = 0.0f;
    for (uint t = 0; t < p.taps; t++) {
        uint sy = min((uint)max((int)(g.y + t) - (int)r, 0), p.h - 1);
        uint i = sy * p.w + g.x;
        acc += k[t] * df_diff_f(df64{u_hi[i], u_lo[i]}, base);
    }
    df64 o = df_add_f(base, acc);
    mid[g.y * p.w + g.x] = o.hi;
    mid[n + g.y * p.w + g.x] = o.lo;
}

kernel void blur_x(device const float* mid [[buffer(0)]], device const float* k [[buffer(1)]],
                   device float* out [[buffer(2)]], constant BlurP& p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    uint r = p.taps / 2;
    uint n = p.w * p.h;
    uint row = g.y * p.w;
    df64 base = df64{mid[row + g.x], mid[n + row + g.x]};
    float acc = 0.0f;
    for (uint t = 0; t < p.taps; t++) {
        uint sx = min((uint)max((int)(g.x + t) - (int)r, 0), p.w - 1);
        acc += k[t] * df_diff_f(df64{mid[row + sx], mid[n + row + sx]}, base);
    }
    df64 s = df_add_f(base, acc);
    out[row + g.x] = s.hi + s.lo;
}

// ---- warp (field::warp_exact) ----

struct WarpP { uint w; uint h; uint w2; uint h2; int shx; int shy; float sixth_hi; float sixth_lo; };

// the coordinate ci + t clamped to [0, n - 1] as integer part and fraction
inline int clamp_axis(int ci, float t, uint n, thread float& tt) {
    if (ci < 0) { tt = 0.0f; return 0; }
    if (ci >= (int)n - 1) { tt = 0.0f; return (int)n - 1; }
    tt = t;
    return ci;
}

inline int mirror_idx(int i, int n) {
    if (i >= 0 && i < n) return i;
    if (n == 1) return 0;
    int period = 2 * (n - 1);
    int m = i % period;
    if (m < 0) m += period;
    return m >= n ? period - m : m;
}

inline df64 df_exact_prod(float a, float b) {
    float2 p = two_prod(a, b);
    return df64{p.x, p.y};
}

inline void weights_df64(float t, df64 sixth, thread df64* w) {
    df64 u = df_add_f(df_from_f32(1.0f), -t);
    df64 t2 = df_exact_prod(t, t);
    df64 t3 = df_mul_f(t2, t);
    df64 u3 = df_mul(df_mul(u, u), u);
    w[0] = df_mul(u3, sixth);
    w[1] = df_mul(df_add_f(df_sub(df_mul_f(t3, 3.0f), df_mul_f(t2, 6.0f)), 4.0f), sixth);
    w[2] = df_mul(df_add_f(df_add(df_sub(df_mul_f(t2, 3.0f), df_mul_f(t3, 3.0f)),
                                  df_mul_f(df_from_f32(t), 3.0f)), 1.0f), sixth);
    w[3] = df_mul(t3, sixth);
}

kernel void warp_exact(device const float* coeffs [[buffer(0)]], device const float* sx [[buffer(1)]],
                       device const float* sy [[buffer(2)]], device float* out [[buffer(3)]],
                       constant WarpP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.w || g.y >= p.h) return;
    uint i = g.y * p.w + g.x;
    float px = sx[i];
    float py = sy[i];
    if (!(isfinite(px) && isfinite(py))) {
        out[i] = NAN;
        return;
    }
    float fx = floor(px);
    float fy = floor(py);
    // beyond 2^29 the clamp below gives the same result as the exact integer
    int ix = int(clamp(fx, -536870912.0f, 536870912.0f));
    int iy = int(clamp(fy, -536870912.0f, 536870912.0f));
    float tx, ty;
    int cx = clamp_axis((int)g.x + p.shx + ix, px - fx, p.w2, tx);
    int cy = clamp_axis((int)g.y + p.shy + iy, py - fy, p.h2, ty);
    int xi[4];
    int yi[4];
    for (int k = 0; k < 4; k++) {
        xi[k] = mirror_idx(cx + k - 1, (int)p.w2);
        yi[k] = mirror_idx(cy + k - 1, (int)p.h2);
    }
    df64 sixth = df64{p.sixth_hi, p.sixth_lo};
    df64 wx[4];
    df64 wy[4];
    weights_df64(tx, sixth, wx);
    weights_df64(ty, sixth, wy);
    df64 total = df_from_f32(0.0f);
    for (int j = 0; j < 4; j++) {
        device const float* line = coeffs + (uint)yi[j] * p.w2;
        df64 acc = df_mul_f(wx[0], line[xi[0]]);
        for (int k = 1; k < 4; k++) {
            acc = df_add(acc, df_mul_f(wx[k], line[xi[k]]));
        }
        total = df_add(total, df_mul(acc, wy[j]));
    }
    out[i] = total.hi + total.lo;
}
