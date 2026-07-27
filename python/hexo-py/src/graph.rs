//! The batched MantisNet graph builder: positions to collated index arrays.
//!
//! This is the production implementation of the representation that
//! `python/mantisnet/mantisnet/builder.py` defines — same entities, same
//! conventions, same orderings, field for field. The Python builder remains
//! the normative reference and the parity oracle: `tests/test_rust_builder.py`
//! holds every array this module emits exactly equal to the Python path's
//! output, which is the §12.7-style detector a second implementation owes.
//!
//! Orderings that parity depends on (all inherited from the Python builder):
//! windows sort by the packed key `(q·2²¹ + r)·4 + axis`; incidence is
//! window-major then slot; the decoder table is legal-cell-major, then axis,
//! then offset. Positions build in parallel under rayon with the GIL
//! released; assembly into flat arrays is single-threaded and cheap.

use hexo_engine as engine;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

const QSHIFT: i64 = 1 << 21;
const NEAREST_BUCKETS: i64 = 8;
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

const fn slot_class(k: usize) -> i64 {
    (if k < 3 { k } else { 5 - k }) as i64
}

fn pack(c: engine::HexCoord) -> i64 {
    c.q as i64 * QSHIFT + c.r as i64
}

/// One position's graph, indices local to the position.
pub struct Graph {
    stone_own: Vec<i64>,
    stone_qr: Vec<[i32; 2]>,
    window_feat: Vec<i64>,
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

pub fn build(pos: &engine::Position) -> Result<Graph, String> {
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

    // Candidate windows through every stone, deduplicated and ordered by the
    // packed key — the same sort np.unique gives the Python builder.
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
    let mut live_occ = Vec::new();
    let mut live_ref = Vec::new();
    // Sorted, because `candidates` is: the decoder probes it by binary
    // search, which beats a hash map at this size and probe count.
    let mut live_keys: Vec<i64> = Vec::new();
    for &(key, wr, m0, m1) in &candidates {
        if (m0 > 0) == (m1 > 0) {
            continue; // dead; never empty, since it came through a stone
        }
        let mover_mask = if mover == engine::Player::P0 { m0 } else { m1 };
        let colour = (mover_mask == 0) as i64;
        let occ = m0 | m1;
        let rank = PATTERN_RANK[occ as usize] as i64;
        live_keys.push(key);
        window_feat.push(colour * NUM_PATTERNS + rank);
        live_occ.push(occ);
        live_ref.push(wr);
    }

    // Incidence: window-major, slot-ascending, matching np.nonzero row-major.
    let mut inc_stone = Vec::new();
    let mut inc_window = Vec::new();
    let mut inc_class = Vec::new();
    for (w, (&occ, wr)) in live_occ.iter().zip(&live_ref).enumerate() {
        for k in 0..6 {
            if occ >> k & 1 == 1 {
                let cell = wr.window.cell(k);
                inc_stone.push(stone_index[&pack(cell)]);
                inc_window.push(w as i64);
                inc_class.push(slot_class(k));
            }
        }
    }

    // Decoder table: legal-cell-major, then the (axis, offset) order
    // `windows_through` returns, which is the Python builder's flat order.
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
                dec_cell.push(j as i64);
                dec_window.push(w as i64);
                dec_class.push(slot_class(i % 6));
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
pub struct RawBatch {
    pub n_pos: usize,
    pub max_t: usize,
    pub max_w: usize,
    pub stone_own: Vec<i64>,
    pub window_feat: Vec<i64>,
    pub moves_idx: Vec<i64>,
    pub inc_stone: Vec<i64>,
    pub inc_window: Vec<i64>,
    pub inc_class: Vec<i64>,
    pub stone_slot: Vec<i64>,
    pub coords: Vec<i32>,      // (P, max_t, 2) row-major
    pub attn_valid: Vec<bool>, // (P, max_t)
    pub window_slot: Vec<i64>,
    pub value_valid: Vec<bool>, // (P, max_w)
    pub legal_offsets: Vec<i64>,
    pub cell_pos: Vec<i64>,
    pub dec_cell: Vec<i64>,
    pub dec_window: Vec<i64>,
    pub dec_class: Vec<i64>,
    pub bg_cell: Vec<i64>,
    pub bg_bucket: Vec<i64>,
}

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

/// Build every position in parallel, then collate. The caller has already
/// left the GIL.
pub fn build_batch(positions: &[engine::Position]) -> Result<RawBatch, String> {
    let graphs: Vec<Graph> = positions.par_iter().map(build).collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}

/// Replay each game's first `t` placements, then build, in parallel.
pub fn build_batch_prefixes(games: &[Vec<(i16, i16)>], ts: &[usize]) -> Result<RawBatch, String> {
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
            build(&pos)
        })
        .collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}
