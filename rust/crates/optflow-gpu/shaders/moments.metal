
// ---- window moments and the block-anchored local solve ----
// Ports of gpu_model::solve::{moments_of, local_solve_blocks}. The sums over the window are f32
// sequential sums over the taps from 0 (y pass, then x pass); the streaming over bands of rows
// and chunks of columns keeps one accumulator per output and adds in the same order.
//
// Plan buffer (u32): header of 20 words
//   [0] n  [1] n_global  [2] gw_slot  [3] n_combos  [4] off_gmom  [5] off_mslot  [6] off_combo
//   [7] off_lmom  [8..13) first global slot of the 4 sources (Gxx, Gxy, Gyy, Gw), last = n_global
//   [13..17) largest y power of each global source (the global moments read the y-plane
//   tables of moments_global_y/_x instead)
// gmom: (px, py) per global slot; mslot: global slot of M[i][j] at i*n+j; combo: (local source
// 0..2, py) per combo; lmom: (combo, px) for the unknowns 0..n and then for Gcc.

constant uint MB = 16;      // BLOCK

struct GlobalP { uint w; uint h; uint r; uint ntaps; };

// The global moments as two separable passes over the whole image (the y pass of a padded
// column is that of the image column it replicates, so the sums are those of moments_of).
// ytab: per y plane MY_STRIDE words: source plane of `sums`, py, number of slots, then
// (slot, px) for each slot fed by that y plane.
constant uint MY_ROWS = 16;
constant uint MY_STRIDE = 3 + 2 * 5;

// y pass: ybuf[z * n + y * w + x] = sum_t k_py[t] src[clamp(y + t - r)][x], t from 0;
// thread = (x, group of MY_ROWS rows, y plane z)
kernel void moments_global_y(device const float* sums [[buffer(0)]], constant float* taps [[buffer(1)]],
                             device const uint* ytab [[buffer(2)]], device float* ybuf [[buffer(3)]],
                             constant GlobalP& p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint w = p.w, h = p.h, r = p.r, nt = p.ntaps;
    uint n = w * h;
    uint y0 = g.y * MY_ROWS;
    if (g.x >= w || y0 >= h) return;
    device const uint* e = ytab + g.z * MY_STRIDE;
    device const float* src = sums + e[0] * n + g.x;
    constant float* k = taps + e[1] * nt;
    float win[MY_ROWS];
    float acc[MY_ROWS];
    for (uint j = 0; j < MY_ROWS; j++) {
        acc[j] = 0.0f;
        win[j] = src[min(max(y0 + j, r) - r, h - 1) * w];
    }
    for (uint t = 0; t < nt; t++) {
        float kt = k[t];
        for (uint j = 0; j < MY_ROWS; j++) acc[j] += kt * win[j];
        for (uint j = 0; j + 1 < MY_ROWS; j++) win[j] = win[j + 1];
        win[MY_ROWS - 1] = src[min(max(y0 + MY_ROWS + t, r) - r, h - 1) * w];
    }
    device float* o = ybuf + g.z * n + g.x;
    for (uint j = 0; j < MY_ROWS; j++)
        if (y0 + j < h) o[(y0 + j) * w] = acc[j];
}

// x pass: out[slot * n + y * w + x] = sum_t k_px[t] ybuf[z][y][clamp(x + t - r)], t from 0;
// thread = (row y, group of MY_ROWS columns, global slot); ztab: per slot (y plane, px)
kernel void moments_global_x(device const float* ybuf [[buffer(0)]], constant float* taps [[buffer(1)]],
                             device const uint* ztab [[buffer(2)]], device float* out [[buffer(3)]],
                             constant GlobalP& p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint w = p.w, h = p.h, r = p.r, nt = p.ntaps;
    uint n = w * h;
    uint y = g.x, x0 = g.y * MY_ROWS;
    if (y >= h || x0 >= w) return;
    device const float* src = ybuf + ztab[2 * g.z] * n + y * w;
    constant float* k = taps + ztab[2 * g.z + 1] * nt;
    float win[MY_ROWS];
    float acc[MY_ROWS];
    for (uint j = 0; j < MY_ROWS; j++) {
        acc[j] = 0.0f;
        win[j] = src[min(max(x0 + j, r) - r, w - 1)];
    }
    for (uint t = 0; t < nt; t++) {
        float kt = k[t];
        for (uint j = 0; j < MY_ROWS; j++) acc[j] += kt * win[j];
        for (uint j = 0; j + 1 < MY_ROWS; j++) win[j] = win[j + 1];
        win[MY_ROWS - 1] = src[min(max(x0 + MY_ROWS + t, r) - r, w - 1)];
    }
    device float* o = out + g.z * n + y * w;
    for (uint j = 0; j < MY_ROWS; j++)
        if (x0 + j < w) o[x0 + j] = acc[j];
}

// aconf, aridge, tol2: the aperture prototypes of phaseflow::Aperture (0: off) and phase_tol^2
struct SolveP {
    uint w; uint h; uint r; uint ntaps; uint nbx; uint order_n;
    float ridge; float tol; float pi; float aconf; float aridge; float tol2;
};

// The anchor of the block [x0, x1) x [y0, y1): the displacement of its central pixel (0 where
// not finite) and the central differences there s = [dux/dx, duy/dx, dux/dy, duy/dy].
inline void block_anchor(device const float* uxh, device const float* uxl,
                         device const float* uyh, device const float* uyl, uint w, uint h,
                         uint x0, uint x1, uint y0, uint y1, thread df64& ax, thread df64& ay_,
                         thread float* s) {
    uint cx = (x0 + x1 - 1) / 2, cy = (y0 + y1 - 1) / 2;
    uint c = cy * w + cx;
    ax = df64{uxh[c], uxl[c]};
    ay_ = df64{uyh[c], uyl[c]};
    if (!(isfinite(ax.hi) && isfinite(ay_.hi))) { ax = df_from_f32(0.0f); ay_ = df_from_f32(0.0f); }
    uint xa = c + (cx + 1 < w ? 1u : 0u), xb = c - (cx > 0 ? 1u : 0u);
    uint ya_i = c + (cy + 1 < h ? w : 0u), yb = c - (cy > 0 ? w : 0u);
    float sx0 = 0.0f, sx1 = 0.0f, sy0 = 0.0f, sy1 = 0.0f;
    if (xa != xb) {
        float span = (float)(xa - xb);
        sx0 = df_diff_f(df64{uxh[xa], uxl[xa]}, df64{uxh[xb], uxl[xb]}) / span;
        sx1 = df_diff_f(df64{uyh[xa], uyl[xa]}, df64{uyh[xb], uyl[xb]}) / span;
    }
    if (ya_i != yb) {
        float span = (float)((ya_i - yb) / w);
        sy0 = df_diff_f(df64{uxh[ya_i], uxl[ya_i]}, df64{uxh[yb], uxl[yb]}) / span;
        sy1 = df_diff_f(df64{uyh[ya_i], uyl[ya_i]}, df64{uyh[yb], uyl[yb]}) / span;
    }
    s[0] = sx0; s[1] = sx1; s[2] = sy0; s[3] = sy1;
    if (!(isfinite(s[0]) && isfinite(s[1]) && isfinite(s[2]) && isfinite(s[3]))) {
        s[0] = 0.0f; s[1] = 0.0f; s[2] = 0.0f; s[3] = 0.0f;
    }
}

// the three increment planes (Gxc, Gyc, Gcc) at sums index qi for the local displacement
// (vx, vy), as local_solve_blocks
inline void increment_planes(device const float* sums, uint n, uint qi, float vx, float vy,
                             thread float& o0, thread float& o1, thread float& o2) {
    float gxx = sums[qi], gxy = sums[n + qi], gyy = sums[2 * n + qi];
    float hx = sums[3 * n + qi], hy = sums[4 * n + qi], spp = sums[5 * n + qi];
    o0 = hx + (gxx * vx + gxy * vy);
    o1 = hy + (gxy * vx + gyy * vy);
    o2 = (spp + 2.0f * (hx * vx + hy * vy))
       + (vx * (gxx * vx + gxy * vy) + vy * (gxy * vx + gyy * vy));
}

// ---- local moments of the increment form (per block) ----
// Threadgroup = one block of MB x MB outputs, 256 threads. The extended block (bw + 2r columns)
// is taken in chunks of LC columns; per chunk the rows of the increment planes stream through
// threadgroup memory in bands of LL rows (lane g of a column computes row g of the band), and
// lane g accumulates the y pass of output rows 2g, 2g+1 for its column. Then the x pass of the
// chunk adds to one accumulator per (output pixel, local moment). Every sum is the sequential
// sum over the taps from 0 of moments_of. Writes the NL local moments per pixel to `lm` and the
// block's anchor (ax.hi, ax.lo, ay.hi, ay.lo, s[0..4]) to `anchors`.
constant uint LC = 32;
constant uint LL = 8;

template <uint NC, uint NL>
inline void local_moments_impl(device const float* sums, device const float* uxh,
                               device const float* uxl, device const float* uyh,
                               device const float* uyl, constant float* taps,
                               device const uint* plan, device float* lm, device float* anchors,
                               constant SolveP& p, uint tid, uint2 grp,
                               threadgroup float* band, threadgroup float* ya) {
    uint w = p.w, h = p.h, r = p.r, nt = p.ntaps;
    uint n = w * h;
    uint x0 = grp.x * MB, y0 = grp.y * MB;
    uint x1 = min(x0 + MB, w), y1 = min(y0 + MB, h);
    uint bw = x1 - x0, bh = y1 - y0;
    uint ew = bw + 2 * r, eh = bh + 2 * r;
    uint cc = tid % LC, g = tid / LC;
    uint oj = tid / MB, oi = tid % MB;
    device const uint* combo = plan + plan[6];
    device const uint* lmom = plan + plan[7];
    uint csrc[NC], cpy[NC];
    for (uint k = 0; k < NC; k++) { csrc[k] = combo[2 * k]; cpy[k] = combo[2 * k + 1] * nt; }
    uint lc[NL], lpx[NL];
    for (uint k = 0; k < NL; k++) { lc[k] = lmom[2 * k]; lpx[k] = lmom[2 * k + 1] * nt; }

    df64 ax, ay_;
    float s[4];
    block_anchor(uxh, uxl, uyh, uyl, w, h, x0, x1, y0, y1, ax, ay_, s);
    if (tid == 0) {
        device float* a = anchors + 8 * (grp.y * p.nbx + grp.x);
        a[0] = ax.hi; a[1] = ax.lo; a[2] = ay_.hi; a[3] = ay_.lo;
        a[4] = s[0]; a[5] = s[1]; a[6] = s[2]; a[7] = s[3];
    }
    uint cx = (x0 + x1 - 1) / 2, cy = (y0 + y1 - 1) / 2;
    float cxf = (float)cx, cyf = (float)cy, rf = (float)r;

    float accx[NL];
    for (uint k = 0; k < NL; k++) accx[k] = 0.0f;

    for (uint c0 = 0; c0 < ew; c0 += LC) {
        uint cn = min(LC, ew - c0);
        bool col = cc < cn;
        uint ec = c0 + cc;
        uint qx = min(max(x0 + ec, r) - r, w - 1);
        float dx = (((float)x0 + (float)ec) - rf) - cxf;
        float acc[NC][2];
        for (uint k = 0; k < NC; k++) { acc[k][0] = 0.0f; acc[k][1] = 0.0f; }
        uint j0 = 2 * g;
        for (uint q0 = 0; q0 < eh; q0 += LL) {
            uint q = q0 + g;
            if (col && q < eh) {
                uint qy = min(max(y0 + q, r) - r, h - 1);
                float dy = (((float)y0 + (float)q) - rf) - cyf;
                uint qi = qy * w + qx;
                float vx = df_diff_f(df64{uxh[qi], uxl[qi]}, ax) - (s[0] * dx + s[2] * dy);
                float vy = df_diff_f(df64{uyh[qi], uyl[qi]}, ay_) - (s[1] * dx + s[3] * dy);
                float i0, i1, i2;
                increment_planes(sums, n, qi, vx, vy, i0, i1, i2);
                band[(0 * LL + g) * LC + cc] = i0;
                band[(1 * LL + g) * LC + cc] = i1;
                band[(2 * LL + g) * LC + cc] = i2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (col) {
                for (uint b = 0; b < LL; b++) {
                    uint qq = q0 + b;
                    float pl[3] = {band[b * LC + cc], band[(LL + b) * LC + cc],
                                   band[(2 * LL + b) * LC + cc]};
                    for (uint jj = 0; jj < 2; jj++) {
                        uint j = j0 + jj;
                        if (j < bh && qq >= j && qq - j <= 2 * r) {
                            uint t = qq - j;
                            for (uint k = 0; k < NC; k++) {
                                float v = csrc[k] == 0 ? pl[0] : (csrc[k] == 1 ? pl[1] : pl[2]);
                                acc[k][jj] += taps[cpy[k] + t] * v;
                            }
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (col) {
            for (uint k = 0; k < NC; k++)
                for (uint jj = 0; jj < 2; jj++)
                    if (j0 + jj < bh) ya[(k * MB + j0 + jj) * LC + cc] = acc[k][jj];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (oj < bh && oi < bw) {
            // over the taps (uniform across the threads), the chunk's columns c0 + cq = oi + t
            for (uint t = 0; t < nt; t++) {
                uint cq = oi + t - c0;   // wraps when oi + t < c0
                if (oi + t < c0 || cq >= cn) continue;
                for (uint k = 0; k < NL; k++)
                    accx[k] += taps[lpx[k] + t] * ya[(lc[k] * MB + oj) * LC + cq];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (oj < bh && oi < bw) {
        uint pp = (y0 + oj) * w + x0 + oi;
        for (uint k = 0; k < NL; k++) lm[k * n + pp] = accx[k];
    }
}

#define LOCAL_MOMENTS_ARGS \
    device const float* sums [[buffer(0)]], device const float* uxh [[buffer(1)]], \
    device const float* uxl [[buffer(2)]], device const float* uyh [[buffer(3)]], \
    device const float* uyl [[buffer(4)]], constant float* taps [[buffer(5)]], \
    device const uint* plan [[buffer(6)]], device float* lm [[buffer(7)]], \
    device float* anchors [[buffer(8)]], constant SolveP& p [[buffer(9)]], \
    uint tid [[thread_index_in_threadgroup]], uint2 grp [[threadgroup_position_in_grid]]

// order 1: 5 combos, 7 local moments; order 2: 7 combos, 13 local moments
kernel void local_moments_1(LOCAL_MOMENTS_ARGS) {
    threadgroup float band[3 * LL * LC];
    threadgroup float ya[5 * MB * LC];
    local_moments_impl<5, 7>(sums, uxh, uxl, uyh, uyl, taps, plan, lm, anchors, p, tid, grp, band, ya);
}
kernel void local_moments_2(LOCAL_MOMENTS_ARGS) {
    threadgroup float band[3 * LL * LC];
    threadgroup float ya[7 * MB * LC];
    local_moments_impl<7, 13>(sums, uxh, uxl, uyh, uyl, taps, plan, lm, anchors, p, tid, grp, band, ya);
}

// ---- local moments for radii up to LMAXR: anchors, y pass, x pass as separate kernels ----
// The y pass of a column of the extended block slides a window of MB rows of its increment
// plane (computed on the fly) over the taps; the x pass slides along the rows of the y result.
// The sums are those of local_moments (sequential over the taps from 0). ybuf holds per block
// and combo MB rows of LEW columns (plus MB floats of slack at the end).
constant uint LMAXR = 24;
constant uint LEW = MB + 2 * LMAXR;

kernel void block_anchors(device const float* uxh [[buffer(0)]], device const float* uxl [[buffer(1)]],
                          device const float* uyh [[buffer(2)]], device const float* uyl [[buffer(3)]],
                          device float* anchors [[buffer(4)]], constant SolveP& p [[buffer(5)]],
                          uint b [[thread_position_in_grid]]) {
    uint w = p.w, h = p.h;
    uint nby = (h + MB - 1) / MB;
    if (b >= p.nbx * nby) return;
    uint x0 = (b % p.nbx) * MB, y0 = (b / p.nbx) * MB;
    uint x1 = min(x0 + MB, w), y1 = min(y0 + MB, h);
    df64 ax, ay_;
    float s[4];
    block_anchor(uxh, uxl, uyh, uyl, w, h, x0, x1, y0, y1, ax, ay_, s);
    device float* a = anchors + 8 * b;
    a[0] = ax.hi; a[1] = ax.lo; a[2] = ay_.hi; a[3] = ay_.lo;
    a[4] = s[0]; a[5] = s[1]; a[6] = s[2]; a[7] = s[3];
}

// threadgroup = one block, thread = (column of the extended block, combo); the three
// increment planes of LCH rows at a time are computed once into threadgroup memory
constant uint LCH = 16;

kernel void local_y(device const float* sums [[buffer(0)]], device const float* uxh [[buffer(1)]],
                    device const float* uxl [[buffer(2)]], device const float* uyh [[buffer(3)]],
                    device const float* uyl [[buffer(4)]], constant float* taps [[buffer(5)]],
                    device const uint* plan [[buffer(6)]], device const float* anchors [[buffer(7)]],
                    device float* ybuf [[buffer(8)]], constant SolveP& p [[buffer(9)]],
                    uint3 l [[thread_position_in_threadgroup]], uint3 ls [[threads_per_threadgroup]],
                    uint3 grp [[threadgroup_position_in_grid]]) {
    threadgroup float plc[3 * LCH * LEW];
    uint w = p.w, h = p.h, r = p.r, nt = p.ntaps;
    uint n = w * h;
    uint col = l.x, k = l.y, blk = grp.z;
    uint lid = l.y * ls.x + l.x, nthr = ls.x * ls.y;
    uint x0 = (blk % p.nbx) * MB, y0 = (blk / p.nbx) * MB;
    uint x1 = min(x0 + MB, w), y1 = min(y0 + MB, h);
    uint bw = x1 - x0, bh = y1 - y0;
    uint ew = bw + 2 * r;
    uint ncombo = plan[3];
    device const uint* combo = plan + plan[6];
    bool mine = col < ew && k < ncombo;
    uint src = mine ? combo[2 * k] : 0u, cpy = mine ? combo[2 * k + 1] * nt : 0u;
    device const float* a = anchors + 8 * blk;
    df64 ax = df64{a[0], a[1]};
    df64 ay_ = df64{a[2], a[3]};
    float s0 = a[4], s1 = a[5], s2 = a[6], s3 = a[7];
    float cxf = (float)((x0 + x1 - 1) / 2), cyf = (float)((y0 + y1 - 1) / 2), rf = (float)r;
    float win[MB];
    float acc[MB];
    for (uint j = 0; j < MB; j++) {
        acc[j] = 0.0f; win[j] = 0.0f;
    }
    uint rows = MB + nt - 1;
    for (uint q0 = 0; q0 < rows; q0 += LCH) {
        for (uint e = lid; e < LCH * ew; e += nthr) {
            uint i = e / ew, c = e % ew;
            uint q = q0 + i;
            uint qx = min(max(x0 + c, r) - r, w - 1);
            uint qy = min(max(y0 + q, r) - r, h - 1);
            float dx = (((float)x0 + (float)c) - rf) - cxf;
            float dy = (((float)y0 + (float)q) - rf) - cyf;
            uint qi = qy * w + qx;
            float vx = df_diff_f(df64{uxh[qi], uxl[qi]}, ax) - (s0 * dx + s2 * dy);
            float vy = df_diff_f(df64{uyh[qi], uyl[qi]}, ay_) - (s1 * dx + s3 * dy);
            float i0, i1, i2;
            increment_planes(sums, n, qi, vx, vy, i0, i1, i2);
            plc[(0 * LCH + i) * LEW + c] = i0;
            plc[(1 * LCH + i) * LEW + c] = i1;
            plc[(2 * LCH + i) * LEW + c] = i2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mine) {
            for (uint i = 0; i < LCH; i++) {
                uint q = q0 + i;
                if (q >= rows) break;
                float v = plc[(min(src, 2u) * LCH + i) * LEW + col];
                // from row MB on the window holds rows q - MB .. q - 1: tap t = q - MB for
                // every output row
                if (q >= MB) {
                    float kt = taps[cpy + (q - MB)];
                    for (uint j = 0; j < MB; j++) acc[j] += kt * win[j];
                }
                for (uint j = 0; j + 1 < MB; j++) win[j] = win[j + 1];
                win[MB - 1] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!mine) return;
    float kt = taps[cpy + nt - 1];
    for (uint j = 0; j < MB; j++) acc[j] += kt * win[j];
    device float* o = ybuf + ((blk * ncombo + k) * MB) * LEW + col;
    for (uint j = 0; j < bh; j++) o[j * LEW] = acc[j];
}

// thread = (output row, local moment, block)
kernel void local_x(device const float* ybuf [[buffer(0)]], constant float* taps [[buffer(1)]],
                    device const uint* plan [[buffer(2)]], device float* lm [[buffer(3)]],
                    constant SolveP& p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint w = p.w, h = p.h, nt = p.ntaps;
    uint n = w * h;
    uint oj = g.x, k = g.y, blk = g.z;
    uint x0 = (blk % p.nbx) * MB, y0 = (blk / p.nbx) * MB;
    uint bw = min(x0 + MB, w) - x0, bh = min(y0 + MB, h) - y0;
    if (oj >= bh || k > plan[0]) return;
    device const uint* lmom = plan + plan[7];
    device const float* row = ybuf + ((blk * plan[3] + lmom[2 * k]) * MB + oj) * LEW;
    constant float* kp = taps + lmom[2 * k + 1] * nt;
    float win[MB];
    float acc[MB];
    for (uint i = 0; i < MB; i++) { acc[i] = 0.0f; win[i] = row[i]; }
    for (uint t = 0; t < nt; t++) {
        float kt = kp[t];
        for (uint i = 0; i < MB; i++) acc[i] += kt * win[i];
        for (uint i = 0; i + 1 < MB; i++) win[i] = win[i + 1];
        win[MB - 1] = row[MB + t];
    }
    device float* o = lm + k * n + (y0 + oj) * w + x0;
    for (uint i = 0; i < bw; i++) o[i] = acc[i];
}

// Cholesky solve as cholesky_f32 of the model, for N unknowns
template <uint N>
inline bool cholesky_n(thread float (&m)[N][N], thread float* b) {
    bool ok = true;
    for (uint j = 0; j < N; j++) {
        float d = m[j][j];
        for (uint k = 0; k < j; k++) {
            float v = m[j][k];
            d += (-v) * v;
        }
        ok = ok && (isfinite(d) && d > 0.0f);
        float ljj = sqrt(d);
        m[j][j] = ljj;
        for (uint i = j + 1; i < N; i++) {
            float s = m[i][j];
            for (uint k = 0; k < j; k++) s += (-m[i][k]) * m[j][k];
            m[i][j] = s / ljj;
        }
    }
    for (uint i = 0; i < N; i++) {
        float s = b[i];
        for (uint k = 0; k < i; k++) s += (-m[i][k]) * b[k];
        b[i] = s / m[i][i];
    }
    for (int i = (int)N - 1; i >= 0; i--) {
        float s = b[i];
        for (uint k = i + 1; k < N; k++) s += (-m[k][i]) * b[k];
        b[i] = s / m[i][i];
    }
    bool fin = true;
    for (uint i = 0; i < N; i++) fin = fin && isfinite(b[i]);
    return ok && fin;
}

inline float max0(float v) { return isnan(v) ? v : max(v, 0.0f); }

// ---- the systems of the pixels (local_solve_blocks before the median reset) ----
// Threadgroup = one block (256 threads); `lm` and `anchors` from local_moments; writes the
// energy range of the block (min, max, has_nan) to `erange`.
template <uint N>
inline void solve_pixels_impl(device const float* glob, device const float* lm,
                              device const float* anchors, device const float* uxh,
                              device const float* uxl, device const float* uyh,
                              device const float* uyl, device const uint* plan,
                              device float* oxh, device float* oxl, device float* oyh,
                              device float* oyl, device float* energy, device float* resid,
                              device float* slope, device float* erange, constant SolveP& p,
                              uint tid, uint2 grp, threadgroup float* red) {
    uint w = p.w, h = p.h;
    uint n = w * h;
    uint x0 = grp.x * MB, y0 = grp.y * MB;
    uint x1 = min(x0 + MB, w), y1 = min(y0 + MB, h);
    uint bw = x1 - x0, bh = y1 - y0;
    uint oj = tid / MB, oi = tid % MB;
    device const uint* mslot = plan + plan[5];
    device const float* a = anchors + 8 * (grp.y * p.nbx + grp.x);
    df64 ax = df64{a[0], a[1]};
    df64 ay_ = df64{a[2], a[3]};
    float s[4] = {a[4], a[5], a[6], a[7]};
    float cxf = (float)((x0 + x1 - 1) / 2), cyf = (float)((y0 + y1 - 1) / 2);

    float e = 0.0f;
    bool valid = oj < bh && oi < bw;
    if (valid) {
        uint px = x0 + oi, py = y0 + oj;
        uint pp = py * w + px;
        float m00 = glob[mslot[0] * n + pp];
        float m11 = glob[mslot[N + 1] * n + pp];
        float m01 = glob[mslot[N] * n + pp];
        e = m00 + m11;
        energy[pp] = e;
        oxh[pp] = uxh[pp]; oxl[pp] = uxl[pp]; oyh[pp] = uyh[pp]; oyl[pp] = uyl[pp];
        resid[pp] = p.pi;
        for (uint k = 0; k < 4; k++) slope[k * n + pp] = 0.0f;
        float det2 = m00 * m11 - m01 * m01;
        if (det2 > p.tol * e * e) {
            float mat[N][N];
            for (uint i = 0; i < N; i++)
                for (uint j = 0; j < N; j++)
                    mat[i][j] = j <= i ? glob[mslot[i * N + j] * n + pp] : 0.0f;
            float trace = 0.0f;
            for (uint i = 0; i < N; i++) trace += mat[i][i];
            float rho = p.ridge * trace / (float)N;
            for (uint i = 0; i < N; i++) mat[i][i] += rho;
            float dxp = (float)px - cxf, dyp = (float)py - cyf;
            float lam[N];
            for (uint i = 0; i < N; i++) lam[i] = 0.0f;
            lam[0] = s[0] * dxp + s[2] * dyp;
            lam[1] = s[1] * dxp + s[3] * dyp;
            for (uint i = 0; i < 4; i++) lam[2 + i] = s[i];
            float bp[N];
            for (uint i = 0; i < N; i++) bp[i] = lm[i * n + pp];
            float ax_ = ax.hi + ax.lo, ay2 = ay_.hi + ay_.lo;
            float theta[N];
            for (uint i = 0; i < N; i++) {
                float lf = i == 0 ? ax_ + lam[0] : (i == 1 ? ay2 + lam[1] : lam[i]);
                theta[i] = bp[i] - rho * lf;
            }
            // the eigenvalues of the translation block, and the aperture ridge towards the input
            float half_ = 0.5f * (m00 - m11);
            float lmax = 0.5f * (m00 + m11) + sqrt(half_ * half_ + m01 * m01);
            float lmin = det2 / lmax;
            bool ridged = false;
            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, d0 = 0.0f, d1 = 0.0f;
            if (p.aridge > 0.0f) {
                float lift = p.aridge * (lmin + lmax) - lmin;
                if (lift > 0.0f) {
                    float c = lift / (lmax - lmin);
                    r0 = c * (lmax - m00); r1 = -c * m01; r2 = c * (lmax - m11);
                    d0 = df_diff_f(df64{uxh[pp], uxl[pp]}, ax) - lam[0];
                    d1 = df_diff_f(df64{uyh[pp], uyl[pp]}, ay_) - lam[1];
                    mat[0][0] += r0; mat[1][0] += r1; mat[1][1] += r2;
                    theta[0] += r0 * d0 + r1 * d1;
                    theta[1] += r1 * d0 + r2 * d1;
                    ridged = true;
                }
            }
            if (cholesky_n<N>(mat, theta)) {
                float fit = 0.0f;
                for (uint i = 0; i < N; i++) fit += theta[i] * bp[i];
                float lx = ax_ + lam[0], ly = ay2 + lam[1];
                float lt = lx * (lx + theta[0]) + ly * (ly + theta[1]);
                for (uint i = 2; i < N; i++) lt += lam[i] * (lam[i] + theta[i]);
                float z = lm[N * n + pp];
                float sw = glob[plan[2] * n + pp];
                float zr = (z - fit) + rho * lt;
                if (ridged) {
                    float e0 = d0 - theta[0], e1 = d1 - theta[1];
                    zr += e0 * (r0 * theta[0] + r1 * theta[1]) + e1 * (r1 * theta[0] + r2 * theta[1]);
                }
                float rs = sqrt(max0(zr) / (sw + 1e-30f));
                if (p.aconf > 0.0f) {
                    float q = lmin / (lmin + lmax);
                    if (q < p.aconf) rs = sqrt(rs * rs + p.tol2 * log(p.aconf / q));
                }
                df64 nx = df_add_f(df_add_f(ax, lam[0]), theta[0]);
                df64 ny = df_add_f(df_add_f(ay_, lam[1]), theta[1]);
                oxh[pp] = nx.hi; oxl[pp] = nx.lo; oyh[pp] = ny.hi; oyl[pp] = ny.lo;
                resid[pp] = rs;
                slope[pp] = theta[2] + s[0];
                slope[n + pp] = theta[3] + s[1];
                slope[2 * n + pp] = theta[4] + s[2];
                slope[3 * n + pp] = theta[5] + s[3];
            }
        }
    }

    // energy range of the block: (min, max, has_nan)
    red[tid] = (valid && !isnan(e)) ? e : INFINITY;
    red[256 + tid] = (valid && !isnan(e)) ? e : -INFINITY;
    red[512 + tid] = (valid && isnan(e)) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_ = 128; half_ > 0; half_ /= 2) {
        if (tid < half_) {
            red[tid] = min(red[tid], red[tid + half_]);
            red[256 + tid] = max(red[256 + tid], red[256 + tid + half_]);
            red[512 + tid] = max(red[512 + tid], red[512 + tid + half_]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        uint b = grp.y * p.nbx + grp.x;
        erange[3 * b] = red[0]; erange[3 * b + 1] = red[256]; erange[3 * b + 2] = red[512];
    }
}

#define SOLVE_PIXELS_ARGS \
    device const float* glob [[buffer(0)]], device const float* lm [[buffer(1)]], \
    device const float* anchors [[buffer(2)]], device const float* uxh [[buffer(3)]], \
    device const float* uxl [[buffer(4)]], device const float* uyh [[buffer(5)]], \
    device const float* uyl [[buffer(6)]], device const uint* plan [[buffer(7)]], \
    device float* oxh [[buffer(8)]], device float* oxl [[buffer(9)]], \
    device float* oyh [[buffer(10)]], device float* oyl [[buffer(11)]], \
    device float* energy [[buffer(12)]], device float* resid [[buffer(13)]], \
    device float* slope [[buffer(14)]], device float* erange [[buffer(15)]], \
    constant SolveP& p [[buffer(16)]], uint tid [[thread_index_in_threadgroup]], \
    uint2 grp [[threadgroup_position_in_grid]]

kernel void solve_pixels_1(SOLVE_PIXELS_ARGS) {
    threadgroup float red[768];
    solve_pixels_impl<6>(glob, lm, anchors, uxh, uxl, uyh, uyl, plan, oxh, oxl, oyh, oyl,
                         energy, resid, slope, erange, p, tid, grp, red);
}
kernel void solve_pixels_2(SOLVE_PIXELS_ARGS) {
    threadgroup float red[768];
    solve_pixels_impl<12>(glob, lm, anchors, uxh, uxl, uyh, uyl, plan, oxh, oxl, oyh, oyl,
                          energy, resid, slope, erange, p, tid, grp, red);
}

struct ResetP { uint n; float threshold; float pi; float pad0; };

kernel void reset_below(device const float* energy [[buffer(0)]],
                        device const float* uxh [[buffer(1)]], device const float* uxl [[buffer(2)]],
                        device const float* uyh [[buffer(3)]], device const float* uyl [[buffer(4)]],
                        device float* oxh [[buffer(5)]], device float* oxl [[buffer(6)]],
                        device float* oyh [[buffer(7)]], device float* oyl [[buffer(8)]],
                        device float* resid [[buffer(9)]], device float* slope [[buffer(10)]],
                        constant ResetP& p [[buffer(11)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    if (!(energy[i] > p.threshold)) {
        oxh[i] = uxh[i]; oxl[i] = uxl[i]; oyh[i] = uyh[i]; oyl[i] = uyl[i];
        resid[i] = p.pi;
        for (uint k = 0; k < 4; k++) slope[k * p.n + i] = 0.0f;
    }
}
