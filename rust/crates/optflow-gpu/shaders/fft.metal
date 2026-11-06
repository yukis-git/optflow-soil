// ---- FFT: one threadgroup per line, Stockham auto-sort stages in threadgroup memory ----
// Concatenated after common.metal (pragmas, metal_stdlib). The line (at most FFT_MAX_LEN complex
// values) lives in `sh`; each stage has at least as many threads as butterflies, so a thread reads
// its r inputs into registers, the group synchronises, and the outputs overwrite the same array.

constant uint FFT_MAX_LEN = 4096;
constant uint FFT_MAX_STAGES = 12;

struct FftParams {
    uint nlen;     // elements per line in the data buffer
    uint len;      // transform length (== nlen, or the padded power of two for Bluestein)
    uint nst;      // number of stages
    uint estride;  // distance between elements of a line, in complex values
    uint lstride;  // distance between lines
    uint pstride;  // distance between planes
    uint nlines;
    float sgn;     // +1 forward (exp(-i..)), -1 inverse
    float scale;   // the last write of the pass multiplies by it (1 leaves the values exact)
    uint r[FFT_MAX_STAGES];
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// exp(-2 pi i idx / len) for the forward direction, conjugated for sg = -1
inline float2 fft_root(device const float2* tw, uint idx, float sg) {
    float2 t = tw[idx];
    return float2(t.x, t.y * sg);
}

// -i*v for sg = +1, +i*v for sg = -1
inline float2 fft_rot(float2 v, float sg) { return float2(sg * v.y, -sg * v.x); }

// x * exp(-i pi/4 sg) and x * exp(-3 i pi/4 sg)
inline float2 fft_w8_1(float2 x, float sg) {
    const float c = 0.70710678118654752f;
    return float2((x.x + sg * x.y) * c, (x.y - sg * x.x) * c);
}
inline float2 fft_w8_3(float2 x, float sg) {
    const float c = 0.70710678118654752f;
    return float2((sg * x.y - x.x) * c, -(x.y + sg * x.x) * c);
}

// the r inputs of butterfly `tid` of a stage, multiplied by their twiddles
template <uint R>
inline void fft_load(threadgroup float2* sh, device const float2* tw, uint tid, uint nb, uint k,
                     uint twstep, float sg, thread float2* v) {
    for (uint t = 0; t < R; t++) {
        float2 x = sh[tid + t * nb];
        if (k > 0 && t > 0) x = cmul(x, fft_root(tw, t * k * twstep, sg));
        v[t] = x;
    }
}

inline void fft_stages(threadgroup float2* sh, device const float2* tw, constant FftParams& p,
                       uint tid, float sg) {
    uint len = p.len;
    uint ns = 1;
    for (uint s = 0; s < p.nst; s++) {
        uint r = p.r[s];
        uint nb = len / r;
        bool active = tid < nb;
        uint k = active ? tid % ns : 0;
        uint twstep = len / (ns * r);
        uint base = (tid / ns) * ns * r + k;
        if (r == 8) {
            float2 v[8];
            if (active) fft_load<8>(sh, tw, tid, nb, k, twstep, sg, v);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                float2 a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
                float2 b0 = v[0] - v[4];
                float2 b1 = fft_w8_1(v[1] - v[5], sg);
                float2 b2 = fft_rot(v[2] - v[6], sg);
                float2 b3 = fft_w8_3(v[3] - v[7], sg);
                float2 pa = a0 + a2, qa = a0 - a2, ra = a1 + a3, sa = fft_rot(a1 - a3, sg);
                float2 pb = b0 + b2, qb = b0 - b2, rb = b1 + b3, sb = fft_rot(b1 - b3, sg);
                sh[base] = pa + ra;
                sh[base + ns] = pb + rb;
                sh[base + 2 * ns] = qa + sa;
                sh[base + 3 * ns] = qb + sb;
                sh[base + 4 * ns] = pa - ra;
                sh[base + 5 * ns] = pb - rb;
                sh[base + 6 * ns] = qa - sa;
                sh[base + 7 * ns] = qb - sb;
            }
        } else if (r == 4) {
            float2 v[4];
            if (active) fft_load<4>(sh, tw, tid, nb, k, twstep, sg, v);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                float2 a = v[0] + v[2];
                float2 b = v[0] - v[2];
                float2 c = v[1] + v[3];
                float2 d = fft_rot(v[1] - v[3], sg);
                sh[base] = a + c;
                sh[base + ns] = b + d;
                sh[base + 2 * ns] = a - c;
                sh[base + 3 * ns] = b - d;
            }
        } else if (r == 2) {
            float2 v[2];
            if (active) fft_load<2>(sh, tw, tid, nb, k, twstep, sg, v);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                sh[base] = v[0] + v[1];
                sh[base + ns] = v[0] - v[1];
            }
        } else if (r == 3) {
            float2 v[3];
            if (active) fft_load<3>(sh, tw, tid, nb, k, twstep, sg, v);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                const float s3 = 0.86602540378443865f;
                float2 t1 = v[1] + v[2];
                float2 t2 = v[0] - 0.5f * t1;
                float2 d = fft_rot(s3 * (v[1] - v[2]), sg);
                sh[base] = v[0] + t1;
                sh[base + ns] = t2 + d;
                sh[base + 2 * ns] = t2 - d;
            }
        } else {
            float2 v[31];
            if (active) {
                for (uint t = 0; t < r; t++) {
                    float2 x = sh[tid + t * nb];
                    if (k > 0 && t > 0) x = cmul(x, fft_root(tw, t * k * twstep, sg));
                    v[t] = x;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                uint rstep = len / r;
                for (uint q = 0; q < r; q++) {
                    float2 acc = v[0];
                    for (uint t = 1; t < r; t++) {
                        acc += cmul(v[t], fft_root(tw, ((t * q) % r) * rstep, sg));
                    }
                    sh[base + q * ns] = acc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ns *= r;
    }
}

inline void fft_lines_direct_impl(device float2* data, device const float2* tw,
                                  constant FftParams& p, uint3 tg, uint tid, uint nthreads,
                                  threadgroup float2* sh) {
    device float2* line = data + tg.z * p.pstride + tg.y * p.lstride;
    for (uint e = tid; e < p.len; e += nthreads) {
        sh[e] = line[e * p.estride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_stages(sh, tw, p, tid, p.sgn);
    for (uint e = tid; e < p.len; e += nthreads) {
        line[e * p.estride] = sh[e] * p.scale;
    }
}

#define FFT_DIRECT_ARGS \
    device float2* data [[buffer(0)]], device const float2* tw [[buffer(1)]], \
    constant FftParams& p [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]], \
    uint3 tpos [[thread_position_in_threadgroup]], uint3 tsize [[threads_per_threadgroup]]

// variants by the threadgroup memory they reserve (it limits the threadgroups per core)
kernel void fft_lines_direct_4k(FFT_DIRECT_ARGS) {
    threadgroup float2 sh[4096];
    fft_lines_direct_impl(data, tw, p, tg, tpos.x, tsize.x, sh);
}
kernel void fft_lines_direct_2k(FFT_DIRECT_ARGS) {
    threadgroup float2 sh[2048];
    fft_lines_direct_impl(data, tw, p, tg, tpos.x, tsize.x, sh);
}
kernel void fft_lines_direct_1k(FFT_DIRECT_ARGS) {
    threadgroup float2 sh[1024];
    fft_lines_direct_impl(data, tw, p, tg, tpos.x, tsize.x, sh);
}

// Bluestein: X_k = c_k * sum_j (x_j c_j) a_{k-j}, c_k = exp(-pi i k^2/n), a = conj(c); the
// convolution is done by FFTs of the padded length `len`. `chirp` holds c (forward); `bf`/`bi`
// the forward FFTs of the filter for the forward/inverse direction. With MUL the line is read as
// spec[e] * filt[e] (`spec` the line of plane 0 of the spectrum, `filt` the real filter of the
// output plane) instead of from `data`.
template <bool MUL>
inline void fft_lines_bluestein_impl(device float2* data, device const float2* tw,
                                     device const float2* chirp, device const float2* bf,
                                     device const float2* bi, device const float2* spec,
                                     device const float* filt, constant FftParams& p, uint3 tg,
                                     uint tid, uint nthreads, threadgroup float2* sh) {
    device float2* line = data + tg.z * p.pstride + tg.y * p.lstride;
    float sg = p.sgn;
    for (uint e = tid; e < p.len; e += nthreads) {
        float2 x = float2(0.0f, 0.0f);
        if (e < p.nlen) {
            float2 c = chirp[e];
            float2 v;
            if constexpr (MUL) {
                uint i = e * p.estride;
                v = spec[tg.y * p.lstride + i] * filt[tg.z * p.pstride + tg.y * p.lstride + i];
            } else {
                v = line[e * p.estride];
            }
            x = cmul(v, float2(c.x, c.y * sg));
        }
        sh[e] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_stages(sh, tw, p, tid, 1.0f);
    device const float2* b = (sg > 0.0f) ? bf : bi;
    for (uint e = tid; e < p.len; e += nthreads) {
        sh[e] = cmul(sh[e], b[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_stages(sh, tw, p, tid, -1.0f);
    float scale = 1.0f / float(p.len);
    for (uint e = tid; e < p.nlen; e += nthreads) {
        float2 c = chirp[e];
        float2 y = cmul(sh[e], float2(c.x, c.y * sg));
        line[e * p.estride] = float2(y.x * scale, y.y * scale) * p.scale;
    }
}

#define FFT_BLUESTEIN_ARGS \
    device float2* data [[buffer(0)]], device const float2* tw [[buffer(1)]], \
    device const float2* chirp [[buffer(2)]], device const float2* bf [[buffer(3)]], \
    device const float2* bi [[buffer(4)]], constant FftParams& p [[buffer(5)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]], \
    uint3 tsize [[threads_per_threadgroup]]

kernel void fft_lines_bluestein_4k(FFT_BLUESTEIN_ARGS) {
    threadgroup float2 sh[4096];
    fft_lines_bluestein_impl<false>(data, tw, chirp, bf, bi, nullptr, nullptr, p, tg, tpos.x,
                                    tsize.x, sh);
}
kernel void fft_lines_bluestein_2k(FFT_BLUESTEIN_ARGS) {
    threadgroup float2 sh[2048];
    fft_lines_bluestein_impl<false>(data, tw, chirp, bf, bi, nullptr, nullptr, p, tg, tpos.x,
                                    tsize.x, sh);
}
kernel void fft_lines_bluestein_1k(FFT_BLUESTEIN_ARGS) {
    threadgroup float2 sh[1024];
    fft_lines_bluestein_impl<false>(data, tw, chirp, bf, bi, nullptr, nullptr, p, tg, tpos.x,
                                    tsize.x, sh);
}

// the rows of the inverse of Fft2Plan::inverse_filtered: read spec * filter (see the impl)
#define FFT_BLUESTEIN_MUL_ARGS \
    device float2* data [[buffer(0)]], device const float2* tw [[buffer(1)]], \
    device const float2* chirp [[buffer(2)]], device const float2* bf [[buffer(3)]], \
    device const float2* bi [[buffer(4)]], device const float2* spec [[buffer(5)]], \
    device const float* filt [[buffer(6)]], constant FftParams& p [[buffer(7)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]], \
    uint3 tsize [[threads_per_threadgroup]]

kernel void fft_lines_bluestein_mul_4k(FFT_BLUESTEIN_MUL_ARGS) {
    threadgroup float2 sh[4096];
    fft_lines_bluestein_impl<true>(data, tw, chirp, bf, bi, spec, filt, p, tg, tpos.x, tsize.x,
                                   sh);
}
kernel void fft_lines_bluestein_mul_2k(FFT_BLUESTEIN_MUL_ARGS) {
    threadgroup float2 sh[2048];
    fft_lines_bluestein_impl<true>(data, tw, chirp, bf, bi, spec, filt, p, tg, tpos.x, tsize.x,
                                   sh);
}
kernel void fft_lines_bluestein_mul_1k(FFT_BLUESTEIN_MUL_ARGS) {
    threadgroup float2 sh[1024];
    fft_lines_bluestein_impl<true>(data, tw, chirp, bf, bi, spec, filt, p, tg, tpos.x, tsize.x,
                                   sh);
}

// ---- FFT with the butterflies in registers (fft_rows_regs, fft_cols_tiled) ----
// The same Stockham stages and twiddle table as `fft_stages`, but every stage is a template on
// its radix, so a thread's R values stay in registers. The first stage reads the line from
// device memory and the last writes it back (no separate copy in and out of threadgroup memory);
// odd prime radices use the conjugate-pair form of the DFT (half the multiplications, the roots
// read once per butterfly).

// In-place R-point DFT of v (y_q = sum_t v_t w^(t q), w = exp(-2 pi i sg / R)); tw[j * rstep]
// is exp(-2 pi i j / R).
template <uint R>
inline void fft_dft(thread float2* v, device const float2* tw, uint rstep, float sg) {
    if constexpr (R == 2) {
        float2 a = v[0];
        v[0] = a + v[1];
        v[1] = a - v[1];
    } else if constexpr (R == 4) {
        float2 a = v[0] + v[2];
        float2 b = v[0] - v[2];
        float2 c = v[1] + v[3];
        float2 d = fft_rot(v[1] - v[3], sg);
        v[0] = a + c;
        v[1] = b + d;
        v[2] = a - c;
        v[3] = b - d;
    } else if constexpr (R == 8) {
        float2 a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
        float2 b0 = v[0] - v[4];
        float2 b1 = fft_w8_1(v[1] - v[5], sg);
        float2 b2 = fft_rot(v[2] - v[6], sg);
        float2 b3 = fft_w8_3(v[3] - v[7], sg);
        float2 pa = a0 + a2, qa = a0 - a2, ra = a1 + a3, sa = fft_rot(a1 - a3, sg);
        float2 pb = b0 + b2, qb = b0 - b2, rb = b1 + b3, sb = fft_rot(b1 - b3, sg);
        v[0] = pa + ra;
        v[1] = pb + rb;
        v[2] = qa + sa;
        v[3] = qb + sb;
        v[4] = pa - ra;
        v[5] = pb - rb;
        v[6] = qa - sa;
        v[7] = qb - sb;
    } else if constexpr (R == 3) {
        const float s3 = 0.86602540378443865f;
        float2 t1 = v[1] + v[2];
        float2 t2 = v[0] - 0.5f * t1;
        float2 d = fft_rot(s3 * (v[1] - v[2]), sg);
        v[0] = v[0] + t1;
        v[1] = t2 + d;
        v[2] = t2 - d;
    } else {
        // odd prime: with a_t = v_t + v_(R-t), b_t = v_t - v_(R-t) (t = 1..H) and w^m = c_m + i s_m,
        // y_q = A_q + i B_q and y_(R-q) = A_q - i B_q, A_q = v_0 + sum_t c_(tq) a_t,
        // B_q = sum_t s_(tq) b_t
        constexpr uint H = (R - 1) / 2;
        float c[H + 1], s[H + 1];
        for (uint j = 1; j <= H; j++) {
            float2 w = tw[j * rstep];
            c[j] = w.x;
            s[j] = w.y * sg;
        }
        float2 a[H + 1], b[H + 1];
        for (uint t = 1; t <= H; t++) {
            a[t] = v[t] + v[R - t];
            b[t] = v[t] - v[R - t];
        }
        float2 y0 = v[0];
        for (uint t = 1; t <= H; t++) y0 += a[t];
        for (uint q = 1; q <= H; q++) {
            float2 A = v[0];
            float2 B = float2(0.0f, 0.0f);
            for (uint t = 1; t <= H; t++) {
                uint m = (t * q) % R;
                uint j = (m <= H) ? m : R - m;
                float sm = (m <= H) ? s[j] : -s[j];
                A += c[j] * a[t];
                B += sm * b[t];
            }
            v[q] = float2(A.x - B.y, A.y + B.x);
            v[R - q] = float2(A.x + B.y, A.y - B.x);
        }
        v[0] = y0;
    }
}

// One stage of radix R (ns = product of the earlier radices). Element e of the line is
// line[e * estride] in device memory and sh[e * shs] in threadgroup memory; the first stage reads
// device memory, the last writes it, the others work in threadgroup memory. `live` is false for
// the threads of a tile column past the last line (they only take part in the barriers). With
// MUL the first stage reads in[i] * fin[i] instead of line[i]; the last stage writes the values
// times `scale`.
template <uint R, bool FIRST, bool LAST, bool MUL>
inline void fft_stage_regs(threadgroup float2* sh, uint shs, device float2* line, uint estride,
                           device const float2* in, device const float* fin,
                           device const float2* tw, uint len, uint ns, uint tid, float sg,
                           float scale, bool live) {
    uint nb = len / R;
    bool active = live && tid < nb;
    float2 v[R];
    uint k = tid % ns;
    uint base = (tid / ns) * ns * R + k;
    if (active) {
        uint twstep = len / (ns * R);
        for (uint t = 0; t < R; t++) {
            uint i = tid + t * nb;
            float2 x;
            if constexpr (FIRST && MUL) x = in[i * estride] * fin[i * estride];
            else if constexpr (FIRST) x = line[i * estride];
            else x = sh[i * shs];
            if (!FIRST && k > 0 && t > 0) x = cmul(x, fft_root(tw, t * k * twstep, sg));
            v[t] = x;
        }
    }
    // in place: every read of the stage before any write (a single stage reads and writes the
    // line in device memory)
    if constexpr (!FIRST) threadgroup_barrier(mem_flags::mem_threadgroup);
    else if constexpr (LAST) threadgroup_barrier(mem_flags::mem_device);
    if (active) {
        fft_dft<R>(v, tw, len / R, sg);
        for (uint q = 0; q < R; q++) {
            uint o = base + q * ns;
            if constexpr (LAST) line[o * estride] = v[q] * scale;
            else sh[o * shs] = v[q];
        }
    }
    if constexpr (!LAST) threadgroup_barrier(mem_flags::mem_threadgroup);
}

// the stage of radix r (uniform over the threadgroup). The kernel's registers are those of its
// largest stage, so there are three sets: MAXR 8 (radices 2, 3, 4, 5, 7, 8), 19 (and 11, 13, 17,
// 19), 31 (and 23, 29, 31).
template <uint MAXR, bool FIRST, bool LAST, bool MUL>
inline void fft_stage_any(uint r, threadgroup float2* sh, uint shs, device float2* line,
                          uint estride, device const float2* in, device const float* fin,
                          device const float2* tw, uint len, uint ns, uint tid, float sg,
                          float scale, bool live) {
#define FFT_STAGE(R) \
    case R: \
        if constexpr (R <= MAXR) \
            fft_stage_regs<R, FIRST, LAST, MUL>(sh, shs, line, estride, in, fin, tw, len, ns, \
                                                tid, sg, scale, live); \
        break;
    switch (r) {
        FFT_STAGE(2)
        FFT_STAGE(3)
        FFT_STAGE(4)
        FFT_STAGE(5)
        FFT_STAGE(7)
        FFT_STAGE(8)
        FFT_STAGE(11)
        FFT_STAGE(13)
        FFT_STAGE(17)
        FFT_STAGE(19)
        FFT_STAGE(23)
        FFT_STAGE(29)
        FFT_STAGE(31)
        default: break;
    }
#undef FFT_STAGE
}

// MUL: the line is read as in[e] * fin[e] (see fft_stage_regs)
template <uint MAXR, bool MUL>
inline void fft_line_regs(threadgroup float2* sh, uint shs, device float2* line, uint estride,
                          device const float2* in, device const float* fin,
                          device const float2* tw, constant FftParams& p, uint tid, bool live) {
    uint len = p.len;
    float sg = p.sgn;
    float sc = p.scale;
    uint n = p.nst;
    if (n == 1) {
        fft_stage_any<MAXR, true, true, MUL>(p.r[0], sh, shs, line, estride, in, fin, tw, len, 1,
                                             tid, sg, sc, live);
        return;
    }
    fft_stage_any<MAXR, true, false, MUL>(p.r[0], sh, shs, line, estride, in, fin, tw, len, 1,
                                          tid, sg, sc, live);
    uint ns = p.r[0];
    for (uint st = 1; st + 1 < n; st++) {
        fft_stage_any<MAXR, false, false, false>(p.r[st], sh, shs, line, estride, in, fin, tw,
                                                 len, ns, tid, sg, sc, live);
        ns *= p.r[st];
    }
    fft_stage_any<MAXR, false, true, false>(p.r[n - 1], sh, shs, line, estride, in, fin, tw, len,
                                            ns, tid, sg, sc, live);
}

// One threadgroup per line (rows, or columns read with a stride). Variants by threadgroup memory
// (1k/2k/4k values) and by the largest radix (see fft_stage_any).
#define FFT_ROWS_REGS(NAME, LEN, MAXR) \
    kernel void NAME(FFT_DIRECT_ARGS) { \
        threadgroup float2 sh[LEN]; \
        device float2* line = data + tg.z * p.pstride + tg.y * p.lstride; \
        fft_line_regs<MAXR, false>(sh, 1, line, p.estride, line, nullptr, tw, p, tpos.x, true); \
    }
FFT_ROWS_REGS(fft_rows_regs_r8_1k, 1024, 8)
FFT_ROWS_REGS(fft_rows_regs_r8_2k, 2048, 8)
FFT_ROWS_REGS(fft_rows_regs_r8_4k, 4096, 8)
FFT_ROWS_REGS(fft_rows_regs_r19_1k, 1024, 19)
FFT_ROWS_REGS(fft_rows_regs_r19_2k, 2048, 19)
FFT_ROWS_REGS(fft_rows_regs_r19_4k, 4096, 19)
FFT_ROWS_REGS(fft_rows_regs_r31_1k, 1024, 31)
FFT_ROWS_REGS(fft_rows_regs_r31_2k, 2048, 31)
FFT_ROWS_REGS(fft_rows_regs_r31_4k, 4096, 31)

// the rows of the inverse of Fft2Plan::inverse_filtered: line l of output plane k is the
// transform of spec[l] * filt[k][l] (the spectrum one plane, the real filters one per plane)
#define FFT_ROWS_REGS_MUL(NAME, LEN, MAXR) \
    kernel void NAME(device float2* data [[buffer(0)]], device const float2* tw [[buffer(1)]], \
                     device const float2* spec [[buffer(2)]], \
                     device const float* filt [[buffer(3)]], constant FftParams& p [[buffer(4)]], \
                     uint3 tg [[threadgroup_position_in_grid]], \
                     uint3 tpos [[thread_position_in_threadgroup]]) { \
        threadgroup float2 sh[LEN]; \
        uint off = tg.y * p.lstride; \
        device float2* line = data + tg.z * p.pstride + off; \
        fft_line_regs<MAXR, true>(sh, 1, line, p.estride, spec + off, \
                                  filt + tg.z * p.pstride + off, tw, p, tpos.x, true); \
    }
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r8_1k, 1024, 8)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r8_2k, 2048, 8)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r8_4k, 4096, 8)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r19_1k, 1024, 19)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r19_2k, 2048, 19)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r19_4k, 4096, 19)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r31_1k, 1024, 31)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r31_2k, 2048, 31)
FFT_ROWS_REGS_MUL(fft_rows_regs_mul_r31_4k, 4096, 31)

// tsize.x adjacent lines per threadgroup (columns: lstride 1), so that a read of element e of
// the tile's lines is tsize.x consecutive values; line c of the tile is sh[c + tsize.x * e]
#define FFT_COLS_TILED(NAME, MAXR) \
    kernel void NAME(FFT_DIRECT_ARGS) { \
        threadgroup float2 sh[4096]; \
        uint c = tpos.x; \
        uint l = tg.x * tsize.x + c; \
        bool live = l < p.nlines; \
        device float2* line = data + tg.z * p.pstride + min(l, p.nlines - 1) * p.lstride; \
        fft_line_regs<MAXR, false>(sh + c, tsize.x, line, p.estride, line, nullptr, tw, p, tpos.y, \
                                   live); \
    }
FFT_COLS_TILED(fft_cols_tiled_r8, 8)
FFT_COLS_TILED(fft_cols_tiled_r19, 19)
FFT_COLS_TILED(fft_cols_tiled_r31, 31)

// 32 x 32 tiles of a h x w plane (row-major, `p.estride` = w, `p.lstride` = h) to its w x h
// transpose in `out`, times `p.scale`
kernel void transpose_c32(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]],
                          constant FftParams& p [[buffer(2)]],
                          uint3 gid [[thread_position_in_grid]],
                          uint3 tpos [[thread_position_in_threadgroup]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
    threadgroup float2 t[32][33];
    uint w = p.estride, h = p.lstride;
    device const float2* src = in + gid.z * p.pstride;
    device float2* dst = out + gid.z * p.pstride;
    if (gid.x < w && gid.y < h) t[tpos.y][tpos.x] = src[gid.y * w + gid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ox = tg.y * 32 + tpos.x;  // column of the output (a row of the input)
    uint oy = tg.x * 32 + tpos.y;  // row of the output (a column of the input)
    if (ox < h && oy < w) dst[oy * h + ox] = t[tpos.x][tpos.y] * p.scale;
}

// Fft2Plan::inverse_filtered where the first pass cannot read the product itself:
// out[k][i] = (spec[i] * filt[k][i]) * scale (`p.nlen` values per plane, `p.nlines` planes)
kernel void fft_filter_multiply(device const float2* spec [[buffer(0)]],
                                device const float* filt [[buffer(1)]],
                                device float2* out [[buffer(2)]], constant FftParams& p [[buffer(3)]],
                                uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.nlen || g.y >= p.nlines) return;
    uint o = g.y * p.nlen + g.x;
    out[o] = (spec[g.x] * filt[o]) * p.scale;
}
