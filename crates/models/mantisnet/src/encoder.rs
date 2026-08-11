//! Batched MantisNet graph encoding and collation.
//!
//! Orderings match the Python builder (`mantisnet.builder`): windows sort by
//! `(q*2^21 + r)*4 + axis`; incidence is window-major then slot; the decoder
//! table is legal-cell-major, then axis, then offset. Parity is checked by
//! `python/mantisnet/tests/test_rust_builder.py`.

use hexo_engine as engine;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fmt;

const QSHIFT: i64 = 1 << 21;
const NEAREST_BUCKETS: i64 = 8;
const WIRE_MAGIC: &[u8; 8] = b"MANTIS\x00\x01";
const WIRE_HEADER_LEN: usize = WIRE_MAGIC.len() + 4 + 4 + 6 * 4;

/// Number of reversal-canonical nonempty, nonfull six-cell occupancy patterns.
pub const NUM_PATTERNS: i64 = 34;

/// Rank of each canonical 6-bit occupancy mask (1–5 bits, mod reversal);
/// `-1` for masks that are empty, full, or non-canonical.
const PATTERN_RANK: [i8; 64] = {
    let mut canon = [0u8; 64];
    let mut m = 0usize;
    while m < 64 {
        let mut rev = 0usize;
        let mut k = 0;
        while k < 6 {
            rev |= ((m >> k) & 1) << (5 - k);
            k += 1;
        }
        canon[m] = if rev < m { rev as u8 } else { m as u8 };
        m += 1;
    }
    let mut rank = [-1i8; 64];
    let mut next = 0i8;
    let mut m = 1usize;
    while m < 63 {
        if canon[m] as usize == m {
            rank[m] = next;
            next += 1;
        }
        m += 1;
    }
    let mut m = 1usize;
    while m < 63 {
        rank[m] = rank[canon[m] as usize];
        m += 1;
    }
    rank
};

/// Number of joint window-occupancy/candidate-slot classes the decoder reads.
///
/// Orbits of the `(mask, slot) -> (reverse6(mask), 5 - slot)` involution
/// over nonempty, nonfull masks with an empty slot: 186 pairs, 93 orbits.
pub const DEC_CLASSES: i64 = 93;

/// Number of joint window-occupancy/occupied-slot classes for stone incidence.
///
/// Same involution as [`DEC_CLASSES`], over pairs whose slot holds a stone:
/// 186 pairs, 93 orbits.
pub const OCC_CLASSES: i64 = 93;

/// Orbit table of `(mask, slot)` pairs under the joint reversal, indexed
/// `mask * 6 + slot`.
///
/// Entries are orbit ranks in ascending `(mask, slot)` order for nonempty,
/// nonfull masks where the slot bit matches `occupied`; `-1` elsewhere.
/// `occupied = false` gives the decoder table, `true` the stone-incidence table.
const fn orbit_table(occupied: bool) -> [i8; 64 * 6] {
    let want = occupied as usize;
    let mut table = [-1i8; 64 * 6];
    let mut next = 0i8;
    let mut m = 1usize;
    while m < 63 {
        let mut rev = 0usize;
        let mut k = 0;
        while k < 6 {
            rev |= ((m >> k) & 1) << (5 - k);
            k += 1;
        }
        let mut s = 0usize;
        while s < 6 {
            if (m >> s) & 1 == want {
                if m < rev || (m == rev && s <= 5 - s) {
                    table[m * 6 + s] = next;
                    next += 1;
                } else {
                    table[m * 6 + s] = table[rev * 6 + (5 - s)];
                }
            }
            s += 1;
        }
        m += 1;
    }
    table
}

/// Decoder class of each `(mask, empty candidate slot)` pair.
const DEC_CLASS: [i8; 64 * 6] = orbit_table(false);

/// Stone-incidence class of each `(mask, occupied slot)` pair.
const OCC_CLASS: [i8; 64 * 6] = orbit_table(true);

/// Number of reversal-canonical nonempty ternary window patterns.
///
/// Under the mixed-window scope a slot is empty (0), own (1), or opponent
/// (2): a window is a base-3 pattern over its six slots, digit at `3^k` for
/// slot `k`, mover-relative. 729 patterns fold to 378 orbits under digit
/// reversal (27 palindromes); the empty pattern is unreachable, leaving 377.
pub const TERN_PATTERNS: i64 = 377;

/// Number of ternary joint decoder classes: empty slots of nonempty patterns.
pub const TERN_DEC_CLASSES: i64 = 726;

/// Number of ternary joint incidence classes: occupied slots. With the
/// decoder classes these are the 2184 nonempty-pattern orbits of the joint
/// involution `(pattern, slot) -> (reverse3(pattern), 5 - slot)`; including
/// the empty pattern's three orbits the involution has 2187, asserted in the
/// table constructor.
pub const TERN_OCC_CLASSES: i64 = 1458;

/// Reverse the base-3 digit string of a ternary pattern.
const fn reverse3(p: usize) -> usize {
    let mut rev = 0usize;
    let mut rem = p;
    let mut k = 0;
    while k < 6 {
        rev = rev * 3 + rem % 3;
        rem /= 3;
        k += 1;
    }
    rev
}

/// Rank of each canonical nonempty ternary pattern (0..377); propagated to
/// noncanonical patterns through their reversal; `-1` for the empty pattern.
const TERN_RANK: [i16; 729] = {
    let mut rank = [-1i16; 729];
    let mut next = 0i16;
    let mut orbits = 0i64;
    let mut p = 0usize;
    while p < 729 {
        if reverse3(p) >= p {
            orbits += 1;
            if p > 0 {
                rank[p] = next;
                next += 1;
            }
        }
        p += 1;
    }
    assert!(orbits == 378 && next as i64 == TERN_PATTERNS);
    let mut p = 1usize;
    while p < 729 {
        let rev = reverse3(p);
        if rev < p {
            rank[p] = rank[rev];
        }
        p += 1;
    }
    rank
};

/// The two ternary joint `(pattern, slot)` class tables, indexed
/// `pattern * 6 + slot`.
///
/// One enumeration of the joint involution in ascending `(pattern, slot)`
/// order — 2187 orbits, asserted — re-ranked over each restriction: empty
/// slots of nonempty patterns (`occupied = false`, the decoder table) or
/// occupied slots (`true`, the incidence table). Entries outside the
/// restriction are `-1`.
const fn tern_orbit_table(occupied: bool) -> [i16; 729 * 6] {
    // The shared enumeration: joint orbit ids over every (pattern, slot).
    let mut joint = [-1i32; 729 * 6];
    let mut next = 0i32;
    let mut p = 0usize;
    while p < 729 {
        let rev = reverse3(p);
        let mut s = 0usize;
        while s < 6 {
            if p < rev || (p == rev && s <= 5 - s) {
                joint[p * 6 + s] = next;
                next += 1;
            } else {
                joint[p * 6 + s] = joint[rev * 6 + (5 - s)];
            }
            s += 1;
        }
        p += 1;
    }
    assert!(next == 2187);

    // Re-rank the restriction: each selected orbit's rank is the count of
    // selected orbits with a smaller joint id — an ascending relabel.
    let want = occupied;
    let mut selected = [false; 2187];
    let mut p = 1usize;
    while p < 729 {
        let mut s = 0usize;
        let mut rem = p;
        while s < 6 {
            if rem.is_multiple_of(3) != want {
                selected[joint[p * 6 + s] as usize] = true;
            }
            rem /= 3;
            s += 1;
        }
        p += 1;
    }
    let mut rank_of = [-1i32; 2187];
    let mut count = 0i32;
    let mut orbit = 0usize;
    while orbit < 2187 {
        if selected[orbit] {
            rank_of[orbit] = count;
            count += 1;
        }
        orbit += 1;
    }
    assert!(
        count as i64
            == if want {
                TERN_OCC_CLASSES
            } else {
                TERN_DEC_CLASSES
            }
    );

    let mut table = [-1i16; 729 * 6];
    let mut p = 1usize;
    while p < 729 {
        let mut s = 0usize;
        let mut rem = p;
        while s < 6 {
            if rem.is_multiple_of(3) != want {
                table[p * 6 + s] = rank_of[joint[p * 6 + s] as usize] as i16;
            }
            rem /= 3;
            s += 1;
        }
        p += 1;
    }
    table
}

/// Ternary decoder class of each `(pattern, empty candidate slot)` pair.
const TERN_DEC_CLASS: [i16; 729 * 6] = tern_orbit_table(false);

/// Ternary incidence class of each `(pattern, occupied slot)` pair.
const TERN_OCC_CLASS: [i16; 729 * 6] = tern_orbit_table(true);

/// Powers of three addressing slot digits of a ternary pattern.
const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];

fn pack(c: engine::HexCoord) -> i64 {
    c.q as i64 * QSHIFT + c.r as i64
}

/// One position's graph, indices local to the position.
pub struct Graph {
    stone_own: Vec<i64>,
    stone_qr: Vec<[i32; 2]>,
    window_feat: Vec<i64>,
    window_id: Vec<i64>,
    inc_stone: Vec<i64>,
    inc_window: Vec<i64>,
    inc_class: Vec<i64>,
    n_legal: usize,
    dec_cell: Vec<i64>,
    dec_window: Vec<i64>,
    dec_class: Vec<i64>,
    bg_cell: Vec<i64>,
    bg_bucket: Vec<i64>,
    moves_remaining: u8,
}

/// Error from decoding a MantisNet wire-format position item.
///
/// Includes the batch item index when produced by [`decode_batch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireError {
    item: Option<usize>,
    detail: String,
}

impl WireError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            item: None,
            detail: detail.into(),
        }
    }

    fn at_item(mut self, item: usize) -> Self {
        self.item = Some(item);
        self
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(item) = self.item {
            write!(f, "encoded MantisNet item {item}: {}", self.detail)
        } else {
            write!(f, "encoded MantisNet item: {}", self.detail)
        }
    }
}

impl std::error::Error for WireError {}

#[derive(Clone, Copy)]
struct WireCounts {
    stones: usize,
    windows: usize,
    incidences: usize,
    legal: usize,
    decoder: usize,
    background: usize,
}

impl WireCounts {
    fn from_graph(graph: &Graph) -> Self {
        Self {
            stones: graph.stone_own.len(),
            windows: graph.window_feat.len(),
            incidences: graph.inc_stone.len(),
            legal: graph.n_legal,
            decoder: graph.dec_cell.len(),
            background: graph.bg_cell.len(),
        }
    }

    fn payload_len(self) -> Option<usize> {
        // Per-stone: 8 (own) + 8 (qr). Per-window: 32 (feat + id triple).
        // Per-incidence: 24. Per-decoder: 24. Per-background: 16.
        self.stones
            .checked_mul(16)?
            .checked_add(self.windows.checked_mul(32)?)?
            .checked_add(self.incidences.checked_mul(24)?)?
            .checked_add(self.decoder.checked_mul(24)?)?
            .checked_add(self.background.checked_mul(16)?)
    }
}

struct WireReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> WireReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], WireError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| WireError::new(format!("{field} length overflows usize")))?;
        let value = self.bytes.get(self.cursor..end).ok_or_else(|| {
            WireError::new(format!(
                "truncated {field}: need {len} bytes, have {}",
                self.bytes.len().saturating_sub(self.cursor)
            ))
        })?;
        self.cursor = end;
        Ok(value)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, WireError> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, WireError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(u32::from_le_bytes(bytes))
    }

    fn i32(&mut self, field: &'static str) -> Result<i32, WireError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(i32::from_le_bytes(bytes))
    }

    fn i64(&mut self, field: &'static str) -> Result<i64, WireError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(i64::from_le_bytes(bytes))
    }
}

fn count_u32(count: usize, field: &'static str) -> u32 {
    u32::try_from(count)
        .unwrap_or_else(|_| panic!("MantisNet {field} count {count} exceeds the wire format"))
}

fn append_i32s(out: &mut Vec<u8>, values: impl IntoIterator<Item = i32>) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn append_i64s(out: &mut Vec<u8>, values: &[i64]) {
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// Append one live position in the versioned worker-to-batcher wire format.
///
/// The bytes already in `out` are left untouched. The appended item is:
///
/// ```text
/// magic[8], MODEL_REPR_VERSION:u32,
/// moves_remaining:u8, reserved_zero[3],
/// stones:u32, windows:u32, incidences:u32, legal:u32,
/// decoder:u32, background:u32,
/// stone_own[stones]:i64,
/// stone_qr[stones][2]:i32,
/// window_feat[windows]:i64,
/// window_id[windows][3]:i64,
/// inc_stone[incidences]:i64, inc_window[incidences]:i64,
/// inc_class[incidences]:i64,
/// dec_cell[decoder]:i64, dec_window[decoder]:i64,
/// dec_class[decoder]:i64,
/// bg_cell[background]:i64, bg_bucket[background]:i64
/// ```
///
/// Every integer is little-endian. A terminal position is a caller protocol
/// violation: the engine, not the network, owns terminal outcomes.
pub fn encode_position(position: &engine::Position, out: &mut Vec<u8>) {
    // The wire format speaks the binary scope: the container never runs a
    // mixed-windows model while the Step 12 knob exists, and a bake replaces
    // this scope wholesale under a MODEL_REPR_VERSION bump.
    let graph = build(position, false)
        .unwrap_or_else(|why| panic!("MantisNet encoder refuses position: {why}"));
    let counts = WireCounts::from_graph(&graph);
    let encoded_counts = [
        count_u32(counts.stones, "stone"),
        count_u32(counts.windows, "window"),
        count_u32(counts.incidences, "incidence"),
        count_u32(counts.legal, "legal-cell"),
        count_u32(counts.decoder, "decoder"),
        count_u32(counts.background, "background"),
    ];
    let payload_len = counts
        .payload_len()
        .expect("a buildable MantisNet graph has a representable wire length");
    out.reserve(
        WIRE_HEADER_LEN
            .checked_add(payload_len)
            .expect("a buildable MantisNet graph has a representable wire length"),
    );

    out.extend_from_slice(WIRE_MAGIC);
    out.extend_from_slice(&crate::MODEL_REPR_VERSION.to_le_bytes());
    out.push(graph.moves_remaining);
    out.extend_from_slice(&[0; 3]);
    for count in encoded_counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    append_i64s(out, &graph.stone_own);
    append_i32s(out, graph.stone_qr.iter().flatten().copied());
    append_i64s(out, &graph.window_feat);
    append_i64s(out, &graph.window_id);
    append_i64s(out, &graph.inc_stone);
    append_i64s(out, &graph.inc_window);
    append_i64s(out, &graph.inc_class);
    append_i64s(out, &graph.dec_cell);
    append_i64s(out, &graph.dec_window);
    append_i64s(out, &graph.dec_class);
    append_i64s(out, &graph.bg_cell);
    append_i64s(out, &graph.bg_bucket);
}

fn read_count(
    reader: &mut WireReader<'_>,
    item_len: usize,
    field: &'static str,
) -> Result<usize, WireError> {
    let count = reader.u32(field)? as usize;
    // Cap every count at the item length to prevent oversized allocations.
    if count > item_len {
        return Err(WireError::new(format!(
            "{field} count {count} exceeds item length {item_len}"
        )));
    }
    Ok(count)
}

fn read_i64_vec(
    reader: &mut WireReader<'_>,
    count: usize,
    field: &'static str,
) -> Result<Vec<i64>, WireError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.i64(field)?);
    }
    Ok(values)
}

fn invalid_feature(field: &'static str, index: usize, value: i64) -> WireError {
    WireError::new(format!("{field}[{index}] has invalid feature {value}"))
}

fn validate_indices(values: &[i64], upper: usize, field: &'static str) -> Result<(), WireError> {
    let upper_i64 = i64::try_from(upper)
        .map_err(|_| WireError::new(format!("{field} upper bound exceeds i64")))?;
    for (index, &value) in values.iter().enumerate() {
        if value < 0 || value >= upper_i64 {
            return Err(WireError::new(format!(
                "{field}[{index}] index {value} is outside 0..{upper}"
            )));
        }
    }
    Ok(())
}

fn decode_graph(bytes: &[u8]) -> Result<Graph, WireError> {
    let mut reader = WireReader::new(bytes);
    if reader.take(WIRE_MAGIC.len(), "magic")? != WIRE_MAGIC {
        return Err(WireError::new("wrong magic"));
    }
    let version = reader.u32("MODEL_REPR_VERSION")?;
    if version != crate::MODEL_REPR_VERSION {
        return Err(WireError::new(format!(
            "MODEL_REPR_VERSION {version} does not match {}",
            crate::MODEL_REPR_VERSION
        )));
    }
    let moves_remaining = reader.u8("moves_remaining")?;
    if !matches!(moves_remaining, 1 | 2) {
        return Err(WireError::new(format!(
            "moves_remaining must be 1 or 2, got {moves_remaining}"
        )));
    }
    if reader.take(3, "reserved bytes")? != [0; 3] {
        return Err(WireError::new("reserved bytes are nonzero"));
    }

    let counts = WireCounts {
        stones: read_count(&mut reader, bytes.len(), "stones")?,
        windows: read_count(&mut reader, bytes.len(), "windows")?,
        incidences: read_count(&mut reader, bytes.len(), "incidences")?,
        legal: read_count(&mut reader, bytes.len(), "legal cells")?,
        decoder: read_count(&mut reader, bytes.len(), "decoder incidences")?,
        background: read_count(&mut reader, bytes.len(), "background cells")?,
    };
    if counts.legal == 0 {
        return Err(WireError::new("a live position must have a legal cell"));
    }
    let expected_len = WIRE_HEADER_LEN
        .checked_add(
            counts
                .payload_len()
                .ok_or_else(|| WireError::new("payload length overflows usize"))?,
        )
        .ok_or_else(|| WireError::new("item length overflows usize"))?;
    match bytes.len().cmp(&expected_len) {
        std::cmp::Ordering::Less => {
            return Err(WireError::new(format!(
                "truncated payload: header describes {expected_len} bytes, got {}",
                bytes.len()
            )));
        }
        std::cmp::Ordering::Greater => {
            return Err(WireError::new(format!(
                "trailing bytes: header describes {expected_len} bytes, got {}",
                bytes.len()
            )));
        }
        std::cmp::Ordering::Equal => {}
    }

    let stone_own = read_i64_vec(&mut reader, counts.stones, "stone_own")?;
    for (index, &value) in stone_own.iter().enumerate() {
        if !matches!(value, 0 | 1) {
            return Err(invalid_feature("stone_own", index, value));
        }
    }

    let mut stone_qr = Vec::with_capacity(counts.stones);
    for index in 0..counts.stones {
        let q = reader.i32("stone_qr.q")?;
        let r = reader.i32("stone_qr.r")?;
        let q16 = i16::try_from(q)
            .map_err(|_| WireError::new(format!("stone_qr[{index}].q is out of range: {q}")))?;
        let r16 = i16::try_from(r)
            .map_err(|_| WireError::new(format!("stone_qr[{index}].r is out of range: {r}")))?;
        if !engine::HexCoord::new(q16, r16).is_valid() {
            return Err(WireError::new(format!(
                "stone_qr[{index}] is not a valid engine coordinate: ({q}, {r})"
            )));
        }
        stone_qr.push([q, r]);
    }

    let window_feat = read_i64_vec(&mut reader, counts.windows, "window_feat")?;
    for (index, &value) in window_feat.iter().enumerate() {
        if !(0..2 * NUM_PATTERNS).contains(&value) {
            return Err(invalid_feature("window_feat", index, value));
        }
    }

    let window_id = read_i64_vec(
        &mut reader,
        counts
            .windows
            .checked_mul(3)
            .ok_or_else(|| WireError::new("window_id length overflows usize".to_string()))?,
        "window_id",
    )?;
    for (index, chunk) in window_id.chunks_exact(3).enumerate() {
        if !(0..3).contains(&chunk[0]) {
            return Err(invalid_feature("window_id", index, chunk[0]));
        }
    }

    let inc_stone = read_i64_vec(&mut reader, counts.incidences, "inc_stone")?;
    let inc_window = read_i64_vec(&mut reader, counts.incidences, "inc_window")?;
    let inc_class = read_i64_vec(&mut reader, counts.incidences, "inc_class")?;
    validate_indices(&inc_stone, counts.stones, "inc_stone")?;
    validate_indices(&inc_window, counts.windows, "inc_window")?;
    for (index, &value) in inc_class.iter().enumerate() {
        if !(0..OCC_CLASSES).contains(&value) {
            return Err(invalid_feature("inc_class", index, value));
        }
    }

    let dec_cell = read_i64_vec(&mut reader, counts.decoder, "dec_cell")?;
    let dec_window = read_i64_vec(&mut reader, counts.decoder, "dec_window")?;
    let dec_class = read_i64_vec(&mut reader, counts.decoder, "dec_class")?;
    validate_indices(&dec_cell, counts.legal, "dec_cell")?;
    validate_indices(&dec_window, counts.windows, "dec_window")?;
    for (index, &value) in dec_class.iter().enumerate() {
        if !(0..DEC_CLASSES).contains(&value) {
            return Err(invalid_feature("dec_class", index, value));
        }
    }

    let bg_cell = read_i64_vec(&mut reader, counts.background, "bg_cell")?;
    let bg_bucket = read_i64_vec(&mut reader, counts.background, "bg_bucket")?;
    validate_indices(&bg_cell, counts.legal, "bg_cell")?;
    for (index, &value) in bg_bucket.iter().enumerate() {
        if !(0..NEAREST_BUCKETS).contains(&value) {
            return Err(invalid_feature("bg_bucket", index, value));
        }
    }

    let mut cell_routes = vec![false; counts.legal];
    for &cell in &dec_cell {
        cell_routes[cell as usize] = true;
    }
    for (index, &cell) in bg_cell.iter().enumerate() {
        let routed = &mut cell_routes[cell as usize];
        if *routed {
            return Err(WireError::new(format!(
                "bg_cell[{index}] duplicates an already routed legal cell {cell}"
            )));
        }
        *routed = true;
    }
    if let Some(cell) = cell_routes.iter().position(|&routed| !routed) {
        return Err(WireError::new(format!(
            "legal cell {cell} has neither a decoder nor background route"
        )));
    }

    debug_assert_eq!(reader.cursor, bytes.len());
    Ok(Graph {
        stone_own,
        stone_qr,
        window_feat,
        window_id,
        inc_stone,
        inc_window,
        inc_class,
        n_legal: counts.legal,
        dec_cell,
        dec_window,
        dec_class,
        bg_cell,
        bg_bucket,
        moves_remaining,
    })
}

/// Build one live position's graph with indices local to that position.
///
/// Returns an error for terminal positions.
///
/// `mixed` selects the window scope: `false` keeps live one-colour windows
/// under the binary tables, `true` every nonempty candidate under the
/// ternary tables (Step 12 knob).
pub fn build(pos: &engine::Position, mixed: bool) -> Result<Graph, String> {
    if pos.is_terminal() {
        return Err("terminal position: the builder refuses it".into());
    }
    let mover = pos.current_player();
    let moves_remaining = match pos.phase() {
        engine::TurnPhase::FirstStone => 2,
        engine::TurnPhase::Opening | engine::TurnPhase::SecondStone => 1,
    };

    let stones: Vec<(engine::HexCoord, engine::Player)> = pos.stones().collect();
    let stone_own: Vec<i64> = stones.iter().map(|&(_, p)| (p != mover) as i64).collect();
    let stone_qr: Vec<[i32; 2]> = stones
        .iter()
        .map(|&(c, _)| [c.q as i32, c.r as i32])
        .collect();
    let stone_index: FxHashMap<i64, i64> = stones
        .iter()
        .enumerate()
        .map(|(i, &(c, _))| (pack(c), i as i64))
        .collect();

    let legal: Vec<engine::HexCoord> = pos.legal_actions().map(|a| a.coord()).collect();
    let n_legal = legal.len();

    if stones.is_empty() {
        // Ply 0: one legal cell, background path, the clamp bucket.
        return Ok(Graph {
            stone_own,
            stone_qr,
            window_feat: vec![],
            window_id: vec![],
            inc_stone: vec![],
            inc_window: vec![],
            inc_class: vec![],
            n_legal,
            dec_cell: vec![],
            dec_window: vec![],
            dec_class: vec![],
            bg_cell: (0..n_legal as i64).collect(),
            bg_bucket: vec![NEAREST_BUCKETS - 1; n_legal],
            moves_remaining,
        });
    }

    // Candidate windows through every stone, deduplicated and sorted by packed key.
    let mut candidates: Vec<(i64, engine::WindowRef, u8, u8)> =
        Vec::with_capacity(stones.len() * 18);
    for &(c, _) in &stones {
        for wr in pos.windows_through(c) {
            if !wr.window.start.is_valid() {
                continue;
            }
            let key = pack(wr.window.start) * 4 + wr.window.axis.index() as i64;
            candidates.push((
                key,
                wr,
                wr.mask.mask(engine::Player::P0),
                wr.mask.mask(engine::Player::P1),
            ));
        }
    }
    candidates.sort_unstable_by_key(|&(key, ..)| key);
    candidates.dedup_by_key(|&mut (key, ..)| key);

    let mut window_feat = Vec::new();
    let mut window_id = Vec::new();
    let mut live_occ = Vec::new();
    // Ternary slot patterns of the kept windows; unused under the binary scope.
    let mut patterns: Vec<u16> = Vec::new();
    let mut live_ref = Vec::new();
    // Sorted for binary-search lookup by the decoder.
    let mut live_keys: Vec<i64> = Vec::new();
    for &(key, wr, m0, m1) in &candidates {
        if !mixed && (m0 > 0) == (m1 > 0) {
            continue; // dead; never empty, since it came through a stone
        }
        let (own, opp) = if mover == engine::Player::P0 {
            (m0, m1)
        } else {
            (m1, m0)
        };
        let occ = m0 | m1;
        if mixed {
            let mut pattern = 0u16;
            for (k, &place) in POW3.iter().enumerate() {
                let digit = (own >> k & 1) as u16 + 2 * (opp >> k & 1) as u16;
                pattern += digit * place;
            }
            window_feat.push(TERN_RANK[pattern as usize] as i64);
            patterns.push(pattern);
        } else {
            let colour = (own == 0) as i64;
            let rank = PATTERN_RANK[occ as usize] as i64;
            window_feat.push(colour * NUM_PATTERNS + rank);
        }
        live_keys.push(key);
        window_id.push(wr.window.axis.index() as i64);
        window_id.push(wr.window.start.q as i64);
        window_id.push(wr.window.start.r as i64);
        live_occ.push(occ);
        live_ref.push(wr);
    }

    // Incidence: window-major, slot-ascending.
    let mut inc_stone = Vec::new();
    let mut inc_window = Vec::new();
    let mut inc_class = Vec::new();
    for (w, (&occ, wr)) in live_occ.iter().zip(&live_ref).enumerate() {
        for k in 0..6 {
            if occ >> k & 1 == 1 {
                let cell = wr.window.cell(k);
                inc_stone.push(stone_index[&pack(cell)]);
                inc_window.push(w as i64);
                inc_class.push(if mixed {
                    TERN_OCC_CLASS[patterns[w] as usize * 6 + k] as i64
                } else {
                    OCC_CLASS[occ as usize * 6 + k] as i64
                });
            }
        }
    }

    // Decoder table: legal-cell-major, then (axis, offset) order.
    let mut dec_cell = Vec::new();
    let mut dec_window = Vec::new();
    let mut dec_class = Vec::new();
    let mut bg_cell = Vec::new();
    let mut bg_bucket = Vec::new();
    for (j, &cell) in legal.iter().enumerate() {
        let mut covered = false;
        for (i, wr) in pos.windows_through(cell).into_iter().enumerate() {
            if !wr.window.start.is_valid() {
                continue;
            }
            let key = pack(wr.window.start) * 4 + wr.window.axis.index() as i64;
            if let Ok(w) = live_keys.binary_search(&key) {
                let slot = i % 6;
                let class = if mixed {
                    TERN_DEC_CLASS[patterns[w] as usize * 6 + slot] as i64
                } else {
                    DEC_CLASS[live_occ[w] as usize * 6 + slot] as i64
                };
                assert!(
                    class >= 0,
                    "legal cell {cell:?} sits at slot {slot} of a window whose \
                     occupancy {:06b} already fills it",
                    live_occ[w]
                );
                dec_cell.push(j as i64);
                dec_window.push(w as i64);
                dec_class.push(class);
                covered = true;
            }
        }
        if !covered {
            let nearest = stones
                .iter()
                .map(|&(s, _)| engine::hex_distance(cell, s) as i64)
                .min()
                .expect("stones is nonempty here");
            bg_cell.push(j as i64);
            bg_bucket.push(nearest.min(NEAREST_BUCKETS) - 1);
        }
    }

    Ok(Graph {
        stone_own,
        stone_qr,
        window_feat,
        window_id,
        inc_stone,
        inc_window,
        inc_class,
        n_legal,
        dec_cell,
        dec_window,
        dec_class,
        bg_cell,
        bg_bucket,
        moves_remaining,
    })
}

/// Everything `mantisnet.builder.Batch` holds, as flat vectors plus shapes.
#[derive(Debug, PartialEq, Eq)]
pub struct RawBatch {
    /// Number of positions in the batch.
    pub n_pos: usize,
    /// Padded width of each position's `[token; stones]` table.
    pub max_t: usize,
    /// Padded width of each position's `[token; windows]` table.
    pub max_w: usize,
    /// Stone owner features, relative to each position's mover.
    pub stone_own: Vec<i64>,
    /// Live-window colour and canonical-pattern features.
    pub window_feat: Vec<i64>,
    /// Live-window identities as `(axis, start_q, start_r)` triples, flat in
    /// `(N_w, 3)` row-major layout. Coordinates are position-local; the model
    /// consumes them only through reversal-invariant pair classes.
    pub window_id: Vec<i64>,
    /// `moves_remaining - 1` for each position.
    pub moves_idx: Vec<i64>,
    /// Global stone index for each stone-to-window incidence.
    pub inc_stone: Vec<i64>,
    /// Global window index for each stone-to-window incidence.
    pub inc_window: Vec<i64>,
    /// Reversal-invariant joint occupancy/slot class for each stone-to-window
    /// incidence.
    pub inc_class: Vec<i64>,
    /// Flat padded-table slot occupied by each stone.
    pub stone_slot: Vec<i64>,
    /// Stone coordinates in `(P, max_t, 2)` row-major layout.
    pub coords: Vec<i32>,
    /// Valid rows in the `(P, max_t)` stone-attention table.
    pub attn_valid: Vec<bool>,
    /// Flat padded-table slot occupied by each live window.
    pub window_slot: Vec<i64>,
    /// Valid rows in the `(P, max_w)` value-readout table.
    pub value_valid: Vec<bool>,
    /// CSR offsets delimiting each position's legal cells.
    pub legal_offsets: Vec<i64>,
    /// Position index for each concatenated legal cell.
    pub cell_pos: Vec<i64>,
    /// Global legal-cell index for each decoder incidence.
    pub dec_cell: Vec<i64>,
    /// Global live-window index for each decoder incidence.
    pub dec_window: Vec<i64>,
    /// Reversal-invariant joint occupancy/slot class for each decoder incidence.
    pub dec_class: Vec<i64>,
    /// Global legal-cell indices routed through the background decoder.
    pub bg_cell: Vec<i64>,
    /// Nearest-stone distance bucket for each background legal cell.
    pub bg_bucket: Vec<i64>,
}

/// Collate position-local graphs into one globally indexed ragged batch.
pub fn collate(graphs: &[Graph]) -> RawBatch {
    let p = graphs.len();
    let max_t = graphs.iter().map(|g| g.stone_own.len()).max().unwrap_or(0) + 1;
    let max_w = graphs
        .iter()
        .map(|g| g.window_feat.len())
        .max()
        .unwrap_or(0)
        + 1;

    let mut out = RawBatch {
        n_pos: p,
        max_t,
        max_w,
        stone_own: vec![],
        window_feat: vec![],
        window_id: vec![],
        moves_idx: Vec::with_capacity(p),
        inc_stone: vec![],
        inc_window: vec![],
        inc_class: vec![],
        stone_slot: vec![],
        coords: vec![0; p * max_t * 2],
        attn_valid: vec![false; p * max_t],
        window_slot: vec![],
        value_valid: vec![false; p * max_w],
        legal_offsets: Vec::with_capacity(p + 1),
        cell_pos: vec![],
        dec_cell: vec![],
        dec_window: vec![],
        dec_class: vec![],
        bg_cell: vec![],
        bg_bucket: vec![],
    };

    let (mut stone_off, mut win_off, mut cell_off) = (0i64, 0i64, 0i64);
    out.legal_offsets.push(0);
    for (i, g) in graphs.iter().enumerate() {
        let (ns, nw) = (g.stone_own.len(), g.window_feat.len());
        out.stone_own.extend_from_slice(&g.stone_own);
        out.window_feat.extend_from_slice(&g.window_feat);
        out.window_id.extend_from_slice(&g.window_id);
        out.moves_idx.push(g.moves_remaining as i64 - 1);
        out.inc_stone
            .extend(g.inc_stone.iter().map(|&s| s + stone_off));
        out.inc_window
            .extend(g.inc_window.iter().map(|&w| w + win_off));
        out.inc_class.extend_from_slice(&g.inc_class);
        out.stone_slot
            .extend((0..ns).map(|j| (i * max_t + 1 + j) as i64));
        out.window_slot
            .extend((0..nw).map(|j| (i * max_w + 1 + j) as i64));
        out.attn_valid[i * max_t] = true;
        out.value_valid[i * max_w] = true;
        for (j, qr) in g.stone_qr.iter().enumerate() {
            out.coords[(i * max_t + 1 + j) * 2] = qr[0];
            out.coords[(i * max_t + 1 + j) * 2 + 1] = qr[1];
            out.attn_valid[i * max_t + 1 + j] = true;
        }
        for j in 0..nw {
            out.value_valid[i * max_w + 1 + j] = true;
        }
        cell_off += g.n_legal as i64;
        out.legal_offsets.push(cell_off);
        out.cell_pos
            .extend(std::iter::repeat_n(i as i64, g.n_legal));
        out.dec_cell
            .extend(g.dec_cell.iter().map(|&c| c + cell_off - g.n_legal as i64));
        out.dec_window
            .extend(g.dec_window.iter().map(|&w| w + win_off));
        out.dec_class.extend_from_slice(&g.dec_class);
        out.bg_cell
            .extend(g.bg_cell.iter().map(|&c| c + cell_off - g.n_legal as i64));
        out.bg_bucket.extend_from_slice(&g.bg_bucket);
        stone_off += ns as i64;
        win_off += nw as i64;
    }
    out
}

/// Decode worker-produced position items and collate them into one model batch.
///
/// Each input slice must contain exactly one item written by
/// [`encode_position`]. Unknown representation versions, malformed features
/// or indices, truncation, and trailing bytes are all refused.
pub fn decode_batch<'a>(items: impl IntoIterator<Item = &'a [u8]>) -> Result<RawBatch, WireError> {
    let graphs = items
        .into_iter()
        .enumerate()
        .map(|(item, bytes)| decode_graph(bytes).map_err(|error| error.at_item(item)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(collate(&graphs))
}

/// Build every position in parallel, then collate.
pub fn build_batch(positions: &[engine::Position], mixed: bool) -> Result<RawBatch, String> {
    let graphs: Vec<Graph> = positions
        .par_iter()
        .map(|pos| build(pos, mixed))
        .collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}

/// Replay each game's first `t` placements, then build, in parallel.
pub fn build_batch_prefixes(
    games: &[Vec<(i16, i16)>],
    ts: &[usize],
    mixed: bool,
) -> Result<RawBatch, String> {
    if games.len() != ts.len() {
        return Err("games and ts must have equal length".into());
    }
    let graphs: Vec<Graph> = games
        .par_iter()
        .zip(ts)
        .map(|(moves, &t)| {
            if t > moves.len() {
                return Err(format!(
                    "prefix length {t} exceeds game length {}",
                    moves.len()
                ));
            }
            let actions: Vec<engine::Action> = moves[..t]
                .iter()
                .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
                .collect();
            let pos = engine::Position::replay(&actions).map_err(|e| e.to_string())?;
            build(&pos, mixed)
        })
        .collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(moves: &[(i16, i16)]) -> engine::Position {
        let actions: Vec<_> = moves
            .iter()
            .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
            .collect();
        engine::Position::replay(&actions).expect("legal test position")
    }

    fn encoded(position: &engine::Position) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_position(position, &mut bytes);
        bytes
    }

    #[test]
    fn the_opening_batch_has_only_the_token_and_background_cell() {
        let raw = build_batch(&[engine::Position::new()], false).expect("the opening is live");

        assert_eq!(raw.n_pos, 1);
        assert_eq!(raw.max_t, 1);
        assert_eq!(raw.max_w, 1);
        assert!(raw.stone_own.is_empty());
        assert!(raw.window_feat.is_empty());
        assert_eq!(raw.moves_idx, [0]);
        assert!(raw.inc_stone.is_empty());
        assert!(raw.inc_window.is_empty());
        assert!(raw.inc_class.is_empty());
        assert!(raw.stone_slot.is_empty());
        assert_eq!(raw.coords, [0, 0]);
        assert_eq!(raw.attn_valid, [true]);
        assert!(raw.window_slot.is_empty());
        assert_eq!(raw.value_valid, [true]);
        assert_eq!(raw.legal_offsets, [0, 1]);
        assert_eq!(raw.cell_pos, [0]);
        assert!(raw.dec_cell.is_empty());
        assert!(raw.dec_window.is_empty());
        assert!(raw.dec_class.is_empty());
        assert_eq!(raw.bg_cell, [0]);
        assert_eq!(raw.bg_bucket, [NEAREST_BUCKETS - 1]);
    }

    #[test]
    fn the_mixed_scope_keeps_every_nonempty_candidate_with_ternary_classes() {
        let position = replay(&[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]);
        let binary = build(&position, false).expect("live position");
        let mixed = build(&position, true).expect("live position");

        // Every nonempty candidate window through a stone, deduplicated.
        let mut keys = std::collections::HashSet::new();
        for (c, _) in position.stones() {
            for wr in position.windows_through(c) {
                if wr.window.start.is_valid()
                    && (wr.mask.mask(engine::Player::P0) | wr.mask.mask(engine::Player::P1)) != 0
                {
                    keys.insert(pack(wr.window.start) * 4 + wr.window.axis.index() as i64);
                }
            }
        }
        assert_eq!(mixed.window_feat.len(), keys.len());
        assert!(mixed.window_feat.len() > binary.window_feat.len());

        for &feat in &mixed.window_feat {
            assert!((0..TERN_PATTERNS).contains(&feat));
        }
        for &class in &mixed.inc_class {
            assert!((0..TERN_OCC_CLASSES).contains(&class));
        }
        for &class in &mixed.dec_class {
            assert!((0..TERN_DEC_CLASSES).contains(&class));
        }

        // Same stones, same legal cells; each stone appears in exactly the
        // windows that cover it, so the incidence count is the summed slot
        // occupancy of the kept windows.
        assert_eq!(mixed.stone_own, binary.stone_own);
        assert_eq!(mixed.n_legal, binary.n_legal);
        assert!(mixed.inc_stone.len() > binary.inc_stone.len());
    }

    #[test]
    fn prefix_replay_and_position_build_share_the_same_core() {
        let moves = vec![(0, 0), (1, 0), (2, 0), (0, 1)];
        let actions: Vec<engine::Action> = moves
            .iter()
            .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
            .collect();
        let position = engine::Position::replay(&actions).expect("legal fixture");

        let direct = build_batch(&[position], false).expect("live position");
        let replayed =
            build_batch_prefixes(&[moves], &[actions.len()], false).expect("legal prefix fixture");
        assert_eq!(direct, replayed);
    }

    #[test]
    fn malformed_prefix_requests_are_refused() {
        let unequal =
            build_batch_prefixes(&[vec![(0, 0)]], &[], false).expect_err("lengths must agree");
        assert_eq!(unequal, "games and ts must have equal length");

        let too_long =
            build_batch_prefixes(&[vec![(0, 0)]], &[2], false).expect_err("prefix is too long");
        assert_eq!(too_long, "prefix length 2 exceeds game length 1");
    }

    #[test]
    fn wire_round_trip_matches_direct_build_for_different_position_shapes() {
        let positions = vec![
            engine::Position::new(),
            replay(&[(0, 0)]),
            replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]),
            replay(&[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]),
        ];
        let items: Vec<Vec<u8>> = positions.iter().map(encoded).collect();

        let decoded =
            decode_batch(items.iter().map(Vec::as_slice)).expect("encoder output is valid");
        let direct = build_batch(&positions, false).expect("all positions are live");
        assert_eq!(decoded, direct);
    }

    #[test]
    fn wire_encoder_only_appends_to_the_callers_buffer() {
        let prefix = [0x55, 0xaa, 0x13, 0x37];
        let mut bytes = prefix.to_vec();
        encode_position(&replay(&[(0, 0), (1, 0)]), &mut bytes);

        assert_eq!(&bytes[..prefix.len()], prefix);
        decode_batch(std::iter::once(&bytes[prefix.len()..]))
            .expect("the appended suffix is one complete item");
    }

    #[test]
    fn wire_decoder_rejects_magic_version_truncation_and_trailing_bytes() {
        let opening = engine::Position::new();
        let bytes = encoded(&opening);

        let mut wrong_magic = bytes.clone();
        wrong_magic[0] ^= 0xff;
        assert!(
            decode_batch(std::iter::once(wrong_magic.as_slice()))
                .expect_err("magic is part of the contract")
                .to_string()
                .contains("wrong magic")
        );

        let mut wrong_version = bytes.clone();
        wrong_version[WIRE_MAGIC.len()..WIRE_MAGIC.len() + 4]
            .copy_from_slice(&(crate::MODEL_REPR_VERSION + 1).to_le_bytes());
        assert!(
            decode_batch(std::iter::once(wrong_version.as_slice()))
                .expect_err("versions never fall through")
                .to_string()
                .contains("does not match")
        );

        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(
            decode_batch(std::iter::once(truncated.as_slice()))
                .expect_err("truncation is refused")
                .to_string()
                .contains("truncated payload")
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert!(
            decode_batch(std::iter::once(trailing.as_slice()))
                .expect_err("concatenated or dirty items are refused")
                .to_string()
                .contains("trailing bytes")
        );
    }

    #[test]
    fn wire_decoder_validates_counts_and_features_before_collation() {
        let mut huge_legal_count = encoded(&engine::Position::new());
        let legal_count_offset = WIRE_MAGIC.len() + 4 + 4 + 3 * 4;
        huge_legal_count[legal_count_offset..legal_count_offset + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            decode_batch(std::iter::once(huge_legal_count.as_slice()))
                .expect_err("item-sized count cap must run before allocation")
                .to_string()
                .contains("exceeds item length")
        );

        let mut invalid_owner = encoded(&replay(&[(0, 0)]));
        invalid_owner[WIRE_HEADER_LEN..WIRE_HEADER_LEN + 8].copy_from_slice(&2i64.to_le_bytes());
        assert!(
            decode_batch(std::iter::once(invalid_owner.as_slice()))
                .expect_err("features have a closed domain")
                .to_string()
                .contains("stone_own[0] has invalid feature 2")
        );
    }

    #[test]
    fn wire_decoder_bounds_the_joint_decoder_class() {
        let position = replay(&[(0, 0)]);
        let graph = build(&position, false).expect("a one-stone position builds");
        let counts = WireCounts::from_graph(&graph);
        // dec_class follows stone_own, stone_qr, window_feat, window_id, the
        // three incidence arrays, dec_cell, and dec_window.
        let offset = WIRE_HEADER_LEN
            + 8 * counts.stones
            + 8 * counts.stones
            + 8 * counts.windows
            + 8 * 3 * counts.windows
            + 8 * 3 * counts.incidences
            + 8 * 2 * counts.decoder;

        let build_max = graph
            .dec_class
            .iter()
            .copied()
            .max()
            .expect("entries exist");
        assert!((0..DEC_CLASSES).contains(&build_max));
        for out_of_range in [DEC_CLASSES, -1] {
            let mut bytes = encoded(&position);
            bytes[offset..offset + 8].copy_from_slice(&out_of_range.to_le_bytes());
            assert!(
                decode_batch(std::iter::once(bytes.as_slice()))
                    .expect_err("the class table has 93 rows and no more")
                    .to_string()
                    .contains(&format!("dec_class[0] has invalid feature {out_of_range}"))
            );
        }
    }
}
