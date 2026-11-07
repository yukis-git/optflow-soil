//! The binary container of a checkpoint: a small versioned header, a section table and raw
//! little-endian arrays. One file, no per-array files and no text framing.
//!
//! ```text
//! offset  size
//! 0       8     magic
//! 8       4     format version (u32)
//! 12      4     number of sections (u32)
//! 16      8     checksum of the section table (u64)
//! 24      40*n  section table: id u32, encoding u32, count u64, offset u64, length u64, checksum u64
//! ...           the sections, in table order, without gaps
//! ```
//!
//! The file ends exactly where the last section ends, so a truncated file is found from the table
//! alone; each section has its own checksum, checked as it is read. Sections are streamed in
//! pieces of [`PIECE`] bytes, so writing and reading never hold a second copy of an array.
//!
//! Encodings are lossless by construction and chosen per section: an array whose elements all have
//! the same bits is stored as one element, booleans take one bit each, an integer array that fits
//! in a byte is stored in bytes, and a masked array stores only the elements its mask leaves out.

use std::io::{self, Read, Seek, SeekFrom, Write};

pub const MAGIC: [u8; 8] = *b"OFCKPT\r\n";
/// The version of the layout above and of the sections of [`crate::track`].
pub const FORMAT: u32 = 1;

const HEAD: u64 = 24;
const ENTRY: u64 = 40;
/// Bytes per streamed piece; a multiple of the hash's block.
const PIECE: usize = 1 << 20;

/// A 64-bit checksum over 32-byte blocks (four multiply-rotate lanes). It detects damage, it is
/// not cryptographic. Bytes are fed in pieces whose length is a multiple of 32, except the last.
#[derive(Clone)]
pub struct Sum {
    lanes: [u64; 4],
    len: u64,
}

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;

impl Sum {
    pub fn new() -> Sum {
        Sum {
            lanes: [P1, P2, P1 ^ P2, P1.wrapping_add(P2)],
            len: 0,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.len += bytes.len() as u64;
        let (blocks, rest) = bytes.as_chunks::<32>();
        let mut l = self.lanes;
        let mix = |l: &mut [u64; 4], b: &[u8; 32]| {
            let (words, _) = b.as_chunks::<8>();
            for (lane, w) in l.iter_mut().zip(words) {
                let w = u64::from_le_bytes(*w);
                *lane = (*lane ^ w.wrapping_mul(P2))
                    .rotate_left(31)
                    .wrapping_mul(P1);
            }
        };
        for b in blocks {
            mix(&mut l, b);
        }
        if !rest.is_empty() {
            let mut b = [0u8; 32];
            b[..rest.len()].copy_from_slice(rest);
            mix(&mut l, &b);
        }
        self.lanes = l;
    }

    pub fn finish(&self) -> u64 {
        let mut h = self.len.wrapping_mul(P1);
        for l in self.lanes {
            h = (h ^ l).rotate_left(27).wrapping_mul(P2);
        }
        h ^ (h >> 29)
    }
}

impl Default for Sum {
    fn default() -> Sum {
        Sum::new()
    }
}

/// The checksum of a float array, by bits (for what is re-read from the input on resume).
pub fn sum_f32(a: &[f32]) -> u64 {
    let mut s = Sum::new();
    let mut buf = vec![0u8; PIECE];
    for c in a.chunks(PIECE / 4) {
        let b = &mut buf[..c.len() * 4];
        for (o, v) in b.as_chunks_mut::<4>().0.iter_mut().zip(c) {
            *o = v.to_le_bytes();
        }
        s.update(b);
    }
    s.finish()
}

/// One array to write.
pub enum Data<'a> {
    Bytes(&'a [u8]),
    U64(&'a [u64]),
    /// Stored raw, or as one element when all have the same bits.
    F32(&'a [f32]),
    F64(&'a [f64]),
    /// Stored as one bit each, or as one byte when all are the same.
    Bool(&'a [bool]),
    /// Stored as bytes when every value fits, or as one element when all are the same.
    U32(&'a [u32]),
    /// Only the elements where `mask` is false, in order.
    MaskedF32(&'a [f32], &'a [bool]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Enc {
    Raw = 0,
    Fill = 1,
    Bits = 2,
    Narrow = 3,
    Masked = 4,
}

impl Enc {
    fn of(v: u32) -> Option<Enc> {
        [Enc::Raw, Enc::Fill, Enc::Bits, Enc::Narrow, Enc::Masked]
            .into_iter()
            .find(|e| *e as u32 == v)
    }
}

fn uniform<T: Copy, B: PartialEq>(a: &[T], bits: impl Fn(T) -> B) -> bool {
    match a.first() {
        Some(&f) => {
            let f = bits(f);
            a.iter().all(|&v| bits(v) == f)
        }
        None => false,
    }
}

/// Writes `items` in pieces through `put`.
fn stream<T: Copy, const N: usize>(
    a: &[T],
    enc: impl Fn(T) -> [u8; N],
    put: &mut dyn FnMut(&[u8]) -> io::Result<()>,
    buf: &mut [u8],
) -> io::Result<()> {
    for c in a.chunks(PIECE / N) {
        let b = &mut buf[..c.len() * N];
        for (o, &v) in b.as_chunks_mut::<N>().0.iter_mut().zip(c) {
            *o = enc(v);
        }
        put(b)?;
    }
    Ok(())
}

impl Data<'_> {
    /// Encoding, element count and stored length.
    fn plan(&self) -> (Enc, u64, u64) {
        match *self {
            Data::Bytes(a) => (Enc::Raw, a.len() as u64, a.len() as u64),
            Data::U64(a) => (Enc::Raw, a.len() as u64, a.len() as u64 * 8),
            Data::F32(a) if uniform(a, f32::to_bits) => (Enc::Fill, a.len() as u64, 4),
            Data::F32(a) => (Enc::Raw, a.len() as u64, a.len() as u64 * 4),
            Data::F64(a) if uniform(a, f64::to_bits) => (Enc::Fill, a.len() as u64, 8),
            Data::F64(a) => (Enc::Raw, a.len() as u64, a.len() as u64 * 8),
            Data::Bool(a) if uniform(a, |b| b) => (Enc::Fill, a.len() as u64, 1),
            Data::Bool(a) => (Enc::Bits, a.len() as u64, a.len().div_ceil(8) as u64),
            Data::U32(a) if uniform(a, |v| v) => (Enc::Fill, a.len() as u64, 4),
            Data::U32(a) if a.iter().all(|&v| v <= 255) => {
                (Enc::Narrow, a.len() as u64, a.len() as u64)
            }
            Data::U32(a) => (Enc::Raw, a.len() as u64, a.len() as u64 * 4),
            Data::MaskedF32(a, m) => {
                let n = m.iter().filter(|&&g| !g).count() as u64;
                (Enc::Masked, a.len() as u64, n * 4)
            }
        }
    }

    fn write(
        &self,
        enc: Enc,
        put: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        buf: &mut [u8],
    ) -> io::Result<()> {
        match (self, enc) {
            (Data::Bytes(a), _) => {
                for c in a.chunks(PIECE) {
                    put(c)?;
                }
                Ok(())
            }
            (Data::U64(a), _) => stream(a, u64::to_le_bytes, put, buf),
            (Data::F32(a), Enc::Fill) => put(&a[0].to_le_bytes()),
            (Data::F32(a), _) => stream(a, f32::to_le_bytes, put, buf),
            (Data::F64(a), Enc::Fill) => put(&a[0].to_le_bytes()),
            (Data::F64(a), _) => stream(a, f64::to_le_bytes, put, buf),
            (Data::Bool(a), Enc::Fill) => put(&[u8::from(a[0])]),
            (Data::Bool(a), _) => {
                for c in a.chunks(PIECE * 8) {
                    let b = &mut buf[..c.len().div_ceil(8)];
                    for (o, bits) in b.iter_mut().zip(c.chunks(8)) {
                        let mut v = 0u8;
                        for (i, &on) in bits.iter().enumerate() {
                            v |= u8::from(on) << i;
                        }
                        *o = v;
                    }
                    put(b)?;
                }
                Ok(())
            }
            (Data::U32(a), Enc::Fill) => put(&a[0].to_le_bytes()),
            (Data::U32(a), Enc::Narrow) => stream(a, |v| [v as u8], put, buf),
            (Data::U32(a), _) => stream(a, u32::to_le_bytes, put, buf),
            (Data::MaskedF32(a, m), _) => {
                let mut n = 0;
                for (v, _) in a.iter().zip(*m).filter(|(_, &g)| !g) {
                    buf[n..n + 4].copy_from_slice(&v.to_le_bytes());
                    n += 4;
                    if n == PIECE {
                        put(&buf[..n])?;
                        n = 0;
                    }
                }
                if n > 0 {
                    put(&buf[..n])?;
                }
                Ok(())
            }
        }
    }
}

/// Writes the container to `w`, which is at its start. Returns the bytes written.
pub fn write_container<W: Write + Seek>(w: &mut W, sections: &[(u32, Data)]) -> io::Result<u64> {
    let table_len = ENTRY * sections.len() as u64;
    let mut entries = Vec::with_capacity(sections.len());
    let mut offset = HEAD + table_len;
    let mut buf = vec![0u8; PIECE];
    // the header goes in last, so a file cut short has no valid table
    w.write_all(&vec![0u8; (HEAD + table_len) as usize])?;
    for (id, data) in sections {
        let (enc, count, len) = data.plan();
        let mut sum = Sum::new();
        let mut written = 0u64;
        let mut put = |b: &[u8]| -> io::Result<()> {
            sum.update(b);
            written += b.len() as u64;
            w.write_all(b)
        };
        data.write(enc, &mut put, &mut buf)?;
        debug_assert_eq!(written, len, "section {id}");
        entries.push((*id, enc as u32, count, offset, len, sum.finish()));
        offset += len;
    }
    let mut table = Vec::with_capacity(table_len as usize);
    for (id, enc, count, off, len, sum) in &entries {
        table.extend_from_slice(&id.to_le_bytes());
        table.extend_from_slice(&enc.to_le_bytes());
        table.extend_from_slice(&count.to_le_bytes());
        table.extend_from_slice(&off.to_le_bytes());
        table.extend_from_slice(&len.to_le_bytes());
        table.extend_from_slice(&sum.to_le_bytes());
    }
    let mut ts = Sum::new();
    ts.update(&table);
    w.flush()?;
    w.seek(SeekFrom::Start(0))?;
    w.write_all(&MAGIC)?;
    w.write_all(&FORMAT.to_le_bytes())?;
    w.write_all(&(sections.len() as u32).to_le_bytes())?;
    w.write_all(&ts.finish().to_le_bytes())?;
    w.write_all(&table)?;
    w.flush()?;
    w.seek(SeekFrom::Start(offset))?;
    Ok(offset)
}

/// Why a container cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("not a checkpoint file")]
    Magic,
    #[error("checkpoint format {0} is not supported (this build reads {FORMAT})")]
    Format(u32),
    #[error("the checkpoint is damaged or incomplete: {0}")]
    Damaged(String),
    /// Something that is recomputed on resume did not reproduce what was saved.
    #[error("the checkpoint does not match the input: {0}")]
    Mismatch(&'static str),
}

fn damaged<T>(what: impl Into<String>) -> Result<T, CheckpointError> {
    Err(CheckpointError::Damaged(what.into()))
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    id: u32,
    enc: Enc,
    count: u64,
    offset: u64,
    len: u64,
    sum: u64,
}

/// An opened container: the table was read and checked, the sections are read on demand.
pub struct Reader<R> {
    r: R,
    entries: Vec<Entry>,
}

impl<R: Read + Seek> Reader<R> {
    pub fn open(mut r: R) -> Result<Reader<R>, CheckpointError> {
        let file_len = r.seek(SeekFrom::End(0))?;
        r.seek(SeekFrom::Start(0))?;
        if file_len < HEAD {
            return damaged("too short for a header");
        }
        let mut head = [0u8; HEAD as usize];
        r.read_exact(&mut head)?;
        if head[..8] != MAGIC {
            return Err(CheckpointError::Magic);
        }
        let format = u32::from_le_bytes(head[8..12].try_into().expect("4 bytes"));
        if format != FORMAT {
            return Err(CheckpointError::Format(format));
        }
        let n = u32::from_le_bytes(head[12..16].try_into().expect("4 bytes")) as u64;
        let want = u64::from_le_bytes(head[16..24].try_into().expect("8 bytes"));
        if HEAD + n * ENTRY > file_len {
            return damaged("the section table is cut short");
        }
        let mut table = vec![0u8; (n * ENTRY) as usize];
        r.read_exact(&mut table)?;
        let mut ts = Sum::new();
        ts.update(&table);
        if ts.finish() != want {
            return damaged("the section table has a wrong checksum");
        }
        let mut entries = Vec::with_capacity(n as usize);
        let mut end = HEAD + n * ENTRY;
        for e in table.as_chunks::<{ ENTRY as usize }>().0 {
            let u32_at = |i: usize| u32::from_le_bytes(e[i..i + 4].try_into().expect("4 bytes"));
            let u64_at = |i: usize| u64::from_le_bytes(e[i..i + 8].try_into().expect("8 bytes"));
            let Some(enc) = Enc::of(u32_at(4)) else {
                return damaged("unknown encoding");
            };
            let entry = Entry {
                id: u32_at(0),
                enc,
                count: u64_at(8),
                offset: u64_at(16),
                len: u64_at(24),
                sum: u64_at(32),
            };
            if entry.offset != end {
                return damaged("the sections are not contiguous");
            }
            end = entry
                .offset
                .checked_add(entry.len)
                .ok_or_else(|| CheckpointError::Damaged("section length".into()))?;
            entries.push(entry);
        }
        if end != file_len {
            return damaged(format!("the file has {file_len} bytes, not {end}"));
        }
        Ok(Reader { r, entries })
    }

    fn entry(&self, id: u32) -> Result<Entry, CheckpointError> {
        match self.entries.iter().find(|e| e.id == id) {
            Some(e) => Ok(*e),
            None => damaged(format!("section {id} is missing")),
        }
    }

    pub fn has(&self, id: u32) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// Feeds the stored bytes of a section to `take` in pieces and checks its checksum.
    fn pieces(
        &mut self,
        e: Entry,
        take: &mut dyn FnMut(&[u8]) -> Result<(), CheckpointError>,
    ) -> Result<(), CheckpointError> {
        self.r.seek(SeekFrom::Start(e.offset))?;
        let mut buf = vec![0u8; PIECE.min(e.len as usize)];
        let mut left = e.len as usize;
        let mut sum = Sum::new();
        while left > 0 {
            let n = left.min(PIECE);
            self.r.read_exact(&mut buf[..n])?;
            sum.update(&buf[..n]);
            take(&buf[..n])?;
            left -= n;
        }
        if sum.finish() != e.sum {
            return damaged(format!("section {} has a wrong checksum", e.id));
        }
        Ok(())
    }

    fn fixed<const N: usize, T: Copy>(
        &mut self,
        id: u32,
        count: usize,
        dec: impl Fn([u8; N]) -> T,
        narrow: Option<&dyn Fn(u8) -> T>,
        zero: T,
    ) -> Result<Vec<T>, CheckpointError> {
        let e = self.entry(id)?;
        if e.count != count as u64 {
            return damaged(format!("section {id} has {} elements", e.count));
        }
        let mut out = vec![zero; count];
        match e.enc {
            Enc::Fill => {
                if e.len != N as u64 {
                    return damaged(format!("section {id}: fill of {} bytes", e.len));
                }
                let mut v = zero;
                self.pieces(e, &mut |b| {
                    v = dec(b.try_into().expect("N bytes"));
                    Ok(())
                })?;
                out.fill(v);
            }
            Enc::Raw => {
                if e.len != (count * N) as u64 {
                    return damaged(format!("section {id}: {} bytes", e.len));
                }
                let mut at = 0;
                self.pieces(e, &mut |b| {
                    for (o, c) in out[at..].iter_mut().zip(b.as_chunks::<N>().0) {
                        *o = dec(*c);
                    }
                    at += b.len() / N;
                    Ok(())
                })?;
            }
            Enc::Narrow => {
                let Some(f) = narrow else {
                    return damaged(format!("section {id}: narrowed"));
                };
                if e.len != count as u64 {
                    return damaged(format!("section {id}: {} bytes", e.len));
                }
                let mut at = 0;
                self.pieces(e, &mut |b| {
                    for (o, &c) in out[at..].iter_mut().zip(b) {
                        *o = f(c);
                    }
                    at += b.len();
                    Ok(())
                })?;
            }
            _ => return damaged(format!("section {id}: wrong encoding")),
        }
        Ok(out)
    }

    pub fn bytes(&mut self, id: u32) -> Result<Vec<u8>, CheckpointError> {
        let e = self.entry(id)?;
        let mut out = Vec::with_capacity(e.len as usize);
        self.pieces(e, &mut |b| {
            out.extend_from_slice(b);
            Ok(())
        })?;
        Ok(out)
    }

    pub fn u64s(&mut self, id: u32) -> Result<Vec<u64>, CheckpointError> {
        let n = self.entry(id)?.count as usize;
        self.fixed::<8, u64>(id, n, u64::from_le_bytes, None, 0)
    }

    pub fn f32s(&mut self, id: u32, count: usize) -> Result<Vec<f32>, CheckpointError> {
        self.fixed::<4, f32>(id, count, f32::from_le_bytes, None, 0.0)
    }

    pub fn f64s(&mut self, id: u32, count: usize) -> Result<Vec<f64>, CheckpointError> {
        self.fixed::<8, f64>(id, count, f64::from_le_bytes, None, 0.0)
    }

    pub fn u32s(&mut self, id: u32, count: usize) -> Result<Vec<u32>, CheckpointError> {
        let narrow = |v: u8| u32::from(v);
        self.fixed::<4, u32>(id, count, u32::from_le_bytes, Some(&narrow), 0)
    }

    pub fn bools(&mut self, id: u32, count: usize) -> Result<Vec<bool>, CheckpointError> {
        let e = self.entry(id)?;
        if e.count != count as u64 {
            return damaged(format!("section {id} has {} elements", e.count));
        }
        let mut out = vec![false; count];
        match e.enc {
            Enc::Fill if e.len == 1 => {
                let mut v = false;
                self.pieces(e, &mut |b| {
                    v = b[0] != 0;
                    Ok(())
                })?;
                out.fill(v);
            }
            Enc::Bits if e.len == count.div_ceil(8) as u64 => {
                let mut at = 0;
                self.pieces(e, &mut |b| {
                    for (o, bits) in out[at..].chunks_mut(8).zip(b) {
                        for (i, x) in o.iter_mut().enumerate() {
                            *x = bits >> i & 1 == 1;
                        }
                    }
                    at += b.len() * 8;
                    Ok(())
                })?;
            }
            _ => return damaged(format!("section {id}: wrong encoding")),
        }
        Ok(out)
    }

    /// A [`Data::MaskedF32`] section, the other elements taken from `base`.
    pub fn masked_f32s(
        &mut self,
        id: u32,
        base: &[f32],
        mask: &[bool],
    ) -> Result<Vec<f32>, CheckpointError> {
        let e = self.entry(id)?;
        let stored = mask.iter().filter(|&&g| !g).count() as u64;
        if e.enc != Enc::Masked
            || e.count != base.len() as u64
            || mask.len() != base.len()
            || e.len != stored * 4
        {
            return damaged(format!("section {id}: wrong shape"));
        }
        let mut out = base.to_vec();
        let mut at = 0usize;
        self.pieces(e, &mut |b| {
            for c in b.as_chunks::<4>().0 {
                while mask[at] {
                    at += 1;
                }
                out[at] = f32::from_le_bytes(*c);
                at += 1;
            }
            Ok(())
        })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn lcg(n: usize) -> Vec<u32> {
        let mut s = 1u64;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 32) as u32
            })
            .collect()
    }

    fn build(sections: &[(u32, Data)]) -> Vec<u8> {
        let mut c = Cursor::new(Vec::new());
        let n = write_container(&mut c, sections).unwrap();
        let v = c.into_inner();
        assert_eq!(n, v.len() as u64);
        v
    }

    #[test]
    fn every_encoding_round_trips_bit_for_bit() {
        // more elements than a piece holds, NaN payloads, infinities and -0 among them
        let n = PIECE / 4 + 1000;
        let f32s: Vec<f32> = lcg(n).into_iter().map(f32::from_bits).collect();
        let f64s: Vec<f64> = lcg(n)
            .into_iter()
            .map(|v| f64::from_bits(u64::from(v) << 32 | u64::from(v.rotate_left(7))))
            .collect();
        let bools: Vec<bool> = lcg(n).into_iter().map(|v| v & 4 == 0).collect();
        let wide: Vec<u32> = lcg(n);
        let small: Vec<u32> = lcg(n).into_iter().map(|v| v % 200).collect();
        let nan = vec![f32::from_bits(0x7FC0_1234); n];
        let one = vec![1.0f64; n];
        let on = vec![true; n];
        let zero = vec![7u32; n];
        let bytes = b"abc".to_vec();
        let u64s = vec![1u64, u64::MAX];
        let file = build(&[
            (1, Data::Bytes(&bytes)),
            (2, Data::U64(&u64s)),
            (3, Data::F32(&f32s)),
            (4, Data::F64(&f64s)),
            (5, Data::Bool(&bools)),
            (6, Data::U32(&wide)),
            (7, Data::U32(&small)),
            (8, Data::F32(&nan)),
            (9, Data::F64(&one)),
            (10, Data::Bool(&on)),
            (11, Data::U32(&zero)),
            (12, Data::MaskedF32(&f32s, &bools)),
        ]);
        let mut r = Reader::open(Cursor::new(&file)).unwrap();
        assert_eq!(r.bytes(1).unwrap(), bytes);
        assert_eq!(r.u64s(2).unwrap(), u64s);
        let bits = |a: &[f32]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r.f32s(3, n).unwrap()), bits(&f32s));
        let b64 = |a: &[f64]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(b64(&r.f64s(4, n).unwrap()), b64(&f64s));
        assert_eq!(r.bools(5, n).unwrap(), bools);
        assert_eq!(r.u32s(6, n).unwrap(), wide);
        assert_eq!(r.u32s(7, n).unwrap(), small);
        assert_eq!(bits(&r.f32s(8, n).unwrap()), bits(&nan));
        assert_eq!(b64(&r.f64s(9, n).unwrap()), b64(&one));
        assert_eq!(r.bools(10, n).unwrap(), on);
        assert_eq!(r.u32s(11, n).unwrap(), zero);
        // the elements under the mask come from the base, the others from the file
        let base: Vec<f32> = vec![9.0; n];
        let got = r.masked_f32s(12, &base, &bools).unwrap();
        for i in 0..n {
            let want = if bools[i] { 9.0 } else { f32s[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "{i}");
        }
    }

    #[test]
    fn uniform_arrays_and_masks_take_little_space() {
        let u = vec![0.0f32; 1000];
        let m = vec![true; 1000];
        let g: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let n: Vec<u32> = (0..1000).map(|i| i % 3).collect();
        for (d, len) in [
            (Data::F32(&u), 4),
            (Data::Bool(&m), 1),
            (Data::Bool(&g), 125),
            (Data::U32(&n), 1000),
            (Data::MaskedF32(&u, &m), 0),
        ] {
            assert_eq!(d.plan().2, len);
        }
    }

    #[test]
    fn truncation_and_damage_are_found() {
        let a: Vec<f32> = (0..5000).map(|i| i as f32).collect();
        let file = build(&[(3, Data::F32(&a))]);
        // cut anywhere
        for cut in [0, 7, 23, 30, 63, 64, file.len() - 1] {
            let e = Reader::open(Cursor::new(&file[..cut])).err();
            assert!(e.is_some(), "cut at {cut}");
        }
        // a flipped bit in the data: the section's checksum
        let mut bad = file.clone();
        *bad.last_mut().unwrap() ^= 1;
        let mut r = Reader::open(Cursor::new(&bad)).unwrap();
        assert!(matches!(r.f32s(3, 5000), Err(CheckpointError::Damaged(_))));
        // a flipped bit in the table
        let mut bad = file.clone();
        bad[HEAD as usize + 1] ^= 1;
        assert!(Reader::open(Cursor::new(&bad)).is_err());
        // another version or a foreign file
        let mut bad = file.clone();
        bad[8] = 99;
        assert!(matches!(
            Reader::open(Cursor::new(&bad)),
            Err(CheckpointError::Format(99))
        ));
        let mut bad = file;
        bad[0] = b'x';
        assert!(matches!(
            Reader::open(Cursor::new(&bad)),
            Err(CheckpointError::Magic)
        ));
    }

    #[test]
    fn a_wrong_element_count_is_refused() {
        let a = vec![1.5f32; 10];
        let file = build(&[(3, Data::F32(&a))]);
        let mut r = Reader::open(Cursor::new(&file)).unwrap();
        assert!(r.f32s(3, 11).is_err());
        assert!(r.f32s(4, 10).is_err());
        assert!(r.has(3) && !r.has(4));
    }
}
