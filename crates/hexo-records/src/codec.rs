//! Fixed-width little-endian primitives, and the bounds-checked cursor that reads
//! them back.
//!
//! Nothing here knows what a game is. Every integer is little-endian and its
//! declared width — there are no varints, so a field's size never depends on its
//! value and an offset in an error message is an offset in the file.

use crate::error::RecordError;

/// Append one byte.
pub(crate) fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

/// Append two bytes, little-endian.
pub(crate) fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append two bytes, little-endian, two's complement.
pub(crate) fn put_i16(out: &mut Vec<u8>, value: i16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append four bytes, little-endian.
pub(crate) fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append eight bytes, little-endian.
pub(crate) fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Append a `u16` byte length and then the string's UTF-8 bytes.
///
/// `field` names the string in the error a too-long one raises.
pub(crate) fn put_str(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), RecordError> {
    let len = u16::try_from(value.len()).map_err(|_| RecordError::StringTooLong {
        field,
        len: value.len(),
    })?;
    put_u16(out, len);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

/// A read head over bytes already in memory, reporting absolute file offsets.
///
/// `base` is where `bytes[0]` sits in the file, so an error from a game entry
/// decoded out of a scratch buffer still points at the file.
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    base: u64,
}

impl<'a> Cursor<'a> {
    /// A cursor over `bytes`, which start at `base` in the file.
    pub(crate) const fn new(bytes: &'a [u8], base: u64) -> Self {
        Self {
            bytes,
            pos: 0,
            base,
        }
    }

    /// The file offset of the next unread byte.
    pub(crate) const fn offset(&self) -> u64 {
        self.base + self.pos as u64
    }

    /// How many bytes are left.
    pub(crate) const fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// The next `n` bytes, or [`RecordError::Truncated`] if they are not all there.
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], RecordError> {
        let available = self.remaining();
        if n > available {
            return Err(RecordError::Truncated {
                offset: self.offset(),
                needed: n,
                available,
            });
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// The next byte.
    pub(crate) fn u8(&mut self) -> Result<u8, RecordError> {
        Ok(self.take(1)?[0])
    }

    /// The next two bytes as a `u16`.
    pub(crate) fn u16(&mut self) -> Result<u16, RecordError> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    /// The next two bytes as an `i16`.
    pub(crate) fn i16(&mut self) -> Result<i16, RecordError> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.take(2)?);
        Ok(i16::from_le_bytes(buf))
    }

    /// The next four bytes as a `u32`.
    pub(crate) fn u32(&mut self) -> Result<u32, RecordError> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    /// The next eight bytes as a `u64`.
    pub(crate) fn u64(&mut self) -> Result<u64, RecordError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    /// A `u16` byte length and that many bytes of UTF-8.
    ///
    /// `field` names the string in the errors a short or non-UTF-8 one raises.
    pub(crate) fn string(&mut self, field: &'static str) -> Result<String, RecordError> {
        let len = usize::from(self.u16()?);
        let offset = self.offset();
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| RecordError::Utf8 { field, offset })
    }
}
