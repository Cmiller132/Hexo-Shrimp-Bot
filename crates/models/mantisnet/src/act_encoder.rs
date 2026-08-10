//! MantisNet-ACT position graph construction.
//!
//! The engine remains the only rules implementation: prefix entry points
//! replay through [`hexo_engine::Position`], and graph rows read its canonical
//! stone and legal-action iterators. Representation-specific enumeration and
//! relation tables live here so the Python boundary only converts vectors to
//! NumPy arrays.

use hexo_engine as engine;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

const WINDOW_LEN: usize = 6;
const NUM_AXES: usize = 3;
const POST_ACTION_ROWS: usize = NUM_AXES * WINDOW_LEN;
const TERNARY_CODES: usize = 3usize.pow(WINDOW_LEN as u32);
const CODE_SLOT_PAIRS: usize = TERNARY_CODES * WINDOW_LEN;
const WINDOW_NUMERIC_FEATURES: usize = 5;
const TACTICAL_FEATURES: usize = 12;
const GLOBAL_NUMERIC_FEATURES: usize = 8;
const AUX_COUNT_CAP: usize = 4;
const NEAREST_UNREACHED: i64 = engine::LEGAL_RADIUS as i64 + 1;
const MAX_ORBIT_RADIUS: usize = 12;
const MAX_SAFE_RADIUS: usize = (i16::MAX - engine::COORD_LIMIT) as usize;
const EXPECTED_PATTERN_CLASSES: i64 = 378;
const EXPECTED_CELL_WINDOW_CLASSES: i64 = 2187;
const EXPECTED_POST1_CLASSES: i64 = 729;
const GLOBAL_THREAT_CAP: usize = 8;
const OWN: u8 = 1;
const OPP: u8 = 2;
const EMPTY: u8 = 0;
const OWN_LIVE: u8 = 1;
const OPP_LIVE: u8 = 2;
const MIXED: u8 = 3;
const ALL_OWN_CODE: usize = 364;
const ALL_OPP_CODE: usize = 728;
const AXES: [(i16, i16); NUM_AXES] = [(1, 0), (0, 1), (1, -1)];
const POWERS: [usize; WINDOW_LEN] = [1, 3, 9, 27, 81, 243];

/// Which six-cell windows persist as graph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowScope {
    /// Persist one-colour nonempty windows only.
    Live,
    /// Persist every nonempty window.
    Nonempty,
    /// Persist every window through a stone or legal action, including empty ones.
    ActionRelevant,
}

/// Which board cells persist as graph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellScope {
    /// Persist occupied cells only.
    OccupiedOnly,
    /// Persist occupied and legal cells.
    OccupiedAndLegal,
    /// Persist occupied, legal, and persistent-window cells.
    WindowAndLegal,
}

/// Relation vocabulary used by occupied-to-cell radius edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D6RelationMode {
    /// Exact displacement orbits under all twelve hex-board symmetries.
    Orbit48,
    /// Distance bucket crossed with the on-axis/off-axis flag.
    CoarseDistanceAxis,
}

/// Every MantisNet-ACT configuration field that changes builder output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActBuilderConfig {
    /// Persistent-window scope.
    pub window_scope: WindowScope,
    /// Cell-node scope.
    pub cell_scope: CellScope,
    /// Radius-edge relation vocabulary.
    pub d6_relation_mode: D6RelationMode,
    /// Largest displacement represented by the relation vocabulary.
    pub d_max: usize,
    /// Largest occupied-to-cell edge distance.
    pub occupied_radius: usize,
    /// Whether directed distance-one cell edges are emitted.
    pub use_cell_adjacency: bool,
    /// Whether occupied-to-cell radius edges are emitted.
    pub use_occupied_radius_edges: bool,
    /// Whether the eight global numeric values are emitted.
    pub use_global_numeric_features: bool,
    /// Whether the five per-window numeric values are emitted.
    pub use_window_numeric_features: bool,
    /// Whether the twelve per-action tactical values are emitted.
    pub use_action_tactical_features: bool,
}

impl ActBuilderConfig {
    /// Refuse an internally inconsistent or unsupported builder configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_max == 0 {
            return Err("d_max must be at least 1, got 0".into());
        }
        if self.d6_relation_mode == D6RelationMode::Orbit48 && self.d_max > MAX_ORBIT_RADIUS {
            return Err(format!(
                "d_max must be at most {MAX_ORBIT_RADIUS}, got {}: the exact orbit id band ends before the reserved relations",
                self.d_max
            ));
        }
        if self.occupied_radius > self.d_max {
            return Err(format!(
                "occupied_radius={} exceeds d_max={}",
                self.occupied_radius, self.d_max
            ));
        }
        if self.occupied_radius > MAX_SAFE_RADIUS {
            return Err(format!(
                "occupied_radius={} exceeds the largest radius {MAX_SAFE_RADIUS} that can be added to an engine coordinate without overflow",
                self.occupied_radius
            ));
        }
        if self.use_occupied_radius_edges && self.occupied_radius == 0 {
            return Err(
                "occupied_radius=0 emits no edge; disable use_occupied_radius_edges instead".into(),
            );
        }
        Ok(())
    }

    fn window_numeric_width(&self) -> usize {
        usize::from(self.use_window_numeric_features) * WINDOW_NUMERIC_FEATURES
    }

    fn tactical_width(&self) -> usize {
        usize::from(self.use_action_tactical_features) * TACTICAL_FEATURES
    }

    fn global_numeric_width(&self) -> usize {
        usize::from(self.use_global_numeric_features) * GLOBAL_NUMERIC_FEATURES
    }
}

/// One position's MantisNet-ACT graph, with every index position-local.
#[derive(Debug, PartialEq)]
pub struct ActGraph {
    /// Cell coordinates in strict `(q, r)` order, flat `[cells, 2]`.
    pub cell_qr: Vec<i64>,
    /// Empty/own/opponent cell features.
    pub cell_occupancy: Vec<i64>,
    /// Whether each represented cell is currently legal.
    pub cell_is_legal: Vec<i64>,
    /// Whether each represented cell is occupied.
    pub cell_is_occupied: Vec<i64>,
    /// Nearest-stone distance bucket, with one unreached bucket.
    pub cell_nearest_bucket: Vec<i64>,
    /// Cell index for each engine-ordered legal action, or `-1`.
    pub legal_to_cell_index: Vec<i64>,
    /// Window identities, flat `[windows, 3]` as `(axis, start_q, start_r)`.
    pub window_id: Vec<i64>,
    /// Reversal-orbit class for each ternary window pattern.
    pub window_pattern_class: Vec<i64>,
    /// Empty/own-live/opponent-live/mixed status for each window.
    pub window_status: Vec<i64>,
    /// Native axis for each window.
    pub window_axis: Vec<i64>,
    /// Per-window normalized counts and runs, flat `[windows, width]`.
    pub window_numeric: Vec<f32>,
    /// Width of `window_numeric`, either zero or five.
    pub window_numeric_width: usize,
    /// Cell index for each window slot, flat `[windows, 6]`, or `-1`.
    pub window_cell_index: Vec<i64>,
    /// Joint ternary-pattern/slot class, flat `[windows, 6]`, or `-1`.
    pub window_incidence_class: Vec<i64>,
    /// Whether each window slot has a represented cell, flat `[windows, 6]`.
    pub window_incidence_mask: Vec<bool>,
    /// Source cell of each directed adjacency edge.
    pub adjacency_src: Vec<i64>,
    /// Destination cell of each directed adjacency edge.
    pub adjacency_dst: Vec<i64>,
    /// Undirected axis of each adjacency edge.
    pub adjacency_axis: Vec<i64>,
    /// Occupied source cell of each radius edge.
    pub radius_src: Vec<i64>,
    /// Destination cell of each radius edge.
    pub radius_dst: Vec<i64>,
    /// D6 orbit or coarse relation of each radius edge.
    pub radius_orbit: Vec<i64>,
    /// On-axis route of each radius edge, or `-1` off-axis.
    pub radius_axis_or_neg1: Vec<i64>,
    /// Persistent pre-action window index, flat `[legal, 3, 6]`, or `-1`.
    pub action_window_index: Vec<i64>,
    /// Post-placement joint pattern/slot class, flat `[legal, 3, 6]`.
    pub action_post1_class: Vec<i64>,
    /// Pre-placement window status, flat `[legal, 3, 6]`.
    pub action_pre_status: Vec<i64>,
    /// Per-action tactical values, flat `[legal, width]`.
    pub action_tactical_numeric: Vec<f32>,
    /// Width of `action_tactical_numeric`, either zero or twelve.
    pub action_tactical_numeric_width: usize,
    /// Per-position numeric values.
    pub global_numeric: Vec<f32>,
    /// Width of `global_numeric`, either zero or eight.
    pub global_numeric_width: usize,
    /// Placements left in the current turn.
    pub moves_remaining: u8,
    /// Opening/first/second phase id.
    pub phase_id: u8,
}

impl ActGraph {
    /// Number of represented cells.
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.cell_occupancy.len()
    }

    /// Number of persistent windows.
    #[must_use]
    pub fn n_windows(&self) -> usize {
        self.window_pattern_class.len()
    }

    /// Number of legal actions.
    #[must_use]
    pub fn n_legal(&self) -> usize {
        self.legal_to_cell_index.len()
    }
}

/// Many position graphs concatenated into the model's packed batch frame.
///
/// Every index is shifted by the target family's per-position offset while
/// the sole sentinel, `-1`, remains unchanged. Feature matrices are stored
/// flat together with their fixed width so the Python boundary can expose
/// NumPy arrays without reconstructing any position-local graph.
#[derive(Debug, PartialEq)]
pub struct PackedActBatch {
    /// Number of positions in the batch.
    pub position_count: usize,
    /// CSR offsets for each packed row family.
    pub cell_offsets: Vec<i64>,
    /// CSR offsets for persistent windows.
    pub window_offsets: Vec<i64>,
    /// CSR offsets for legal actions.
    pub legal_offsets: Vec<i64>,
    /// CSR offsets for adjacency edges.
    pub adjacency_offsets: Vec<i64>,
    /// CSR offsets for occupied-radius edges.
    pub radius_offsets: Vec<i64>,
    /// Cell features in graph order.
    pub cell_occupancy: Vec<i64>,
    /// Whether each packed cell is legal.
    pub cell_is_legal: Vec<i64>,
    /// Nearest-stone distance bucket for each packed cell.
    pub cell_nearest_bucket: Vec<i64>,
    /// Engine-ordered legal actions mapped into the global cell frame.
    pub legal_to_cell_index: Vec<i64>,
    /// Window identities and features in each graph's section-seven order.
    pub window_id: Vec<i64>,
    /// Reversal-orbit pattern class for each packed window.
    pub window_pattern_class: Vec<i64>,
    /// Empty/live/mixed status for each packed window.
    pub window_status: Vec<i64>,
    /// Native axis for each packed window.
    pub window_axis: Vec<i64>,
    /// Flat per-window numeric feature rows.
    pub window_numeric: Vec<f32>,
    /// Width of `window_numeric`, either zero or five.
    pub window_numeric_width: usize,
    /// Window incidence in the global cell frame.
    pub window_cell_index: Vec<i64>,
    /// Joint pattern/slot class for each window incidence.
    pub window_incidence_class: Vec<i64>,
    /// Whether each window slot has a represented cell.
    pub window_incidence_mask: Vec<bool>,
    /// Adjacency edges in the global cell frame.
    pub adjacency_src: Vec<i64>,
    /// Destination cell of each packed adjacency edge.
    pub adjacency_dst: Vec<i64>,
    /// Undirected axis of each packed adjacency edge.
    pub adjacency_axis: Vec<i64>,
    /// Occupied-radius edges in the global cell frame.
    pub radius_src: Vec<i64>,
    /// Destination cell of each packed occupied-radius edge.
    pub radius_dst: Vec<i64>,
    /// D6 orbit or coarse relation of each packed radius edge.
    pub radius_orbit: Vec<i64>,
    /// On-axis route of each packed radius edge, or `-1` off-axis.
    pub radius_axis_or_neg1: Vec<i64>,
    /// Counterfactual action rows in the global window frame.
    pub action_window_index: Vec<i64>,
    /// Post-placement joint pattern/slot class for each action row.
    pub action_post1_class: Vec<i64>,
    /// Pre-placement window status for each action row.
    pub action_pre_status: Vec<i64>,
    /// Flat per-action tactical numeric feature rows.
    pub action_tactical_numeric: Vec<f32>,
    /// Width of `action_tactical_numeric`, either zero or twelve.
    pub action_tactical_numeric_width: usize,
    /// One row per position.
    pub phase_id: Vec<i64>,
    /// Placements left in the current turn for each position.
    pub moves_remaining: Vec<i64>,
    /// Flat per-position global numeric feature rows.
    pub global_numeric: Vec<f32>,
    /// Width of `global_numeric`, either zero or eight.
    pub global_numeric_width: usize,
    /// One past the largest radius relation id, or zero for no radius edge.
    pub radius_orbit_bound: i64,
}

/// Section 24.1's six deterministic labels for every legal action.
#[derive(Debug, PartialEq, Eq)]
pub struct ActAuxLabels {
    /// Whether the placement wins immediately.
    pub win_now: Vec<i64>,
    /// Largest own occupancy among the eighteen post-placement windows.
    pub own_max_occupancy: Vec<i64>,
    /// Opponent four/five threats intersected by the placement, capped at four.
    pub opponent_threats_hit: Vec<i64>,
    /// Own five-windows after the placement, capped at four.
    pub own_five_windows_after: Vec<i64>,
    /// Whether some second placement would win.
    pub winning_partner_exists: Vec<i64>,
    /// Distinct winning second placements, capped at four.
    pub winning_partner_count: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct WindowId {
    axis: u8,
    q: i16,
    r: i16,
}

struct PatternTables {
    reverse_code: [u16; TERNARY_CODES],
    pattern_class: [i64; TERNARY_CODES],
    cell_window_class: [i64; CODE_SLOT_PAIRS],
    post1_class: [i64; CODE_SLOT_PAIRS],
    own_count: [u8; TERNARY_CODES],
    opp_count: [u8; TERNARY_CODES],
    empty_count: [u8; TERNARY_CODES],
    own_max_run: [u8; TERNARY_CODES],
    opp_max_run: [u8; TERNARY_CODES],
    status: [u8; TERNARY_CODES],
}

fn digit(code: usize, slot: usize) -> u8 {
    ((code / POWERS[slot]) % 3) as u8
}

fn reverse_code(code: usize) -> usize {
    (0..WINDOW_LEN)
        .map(|slot| digit(code, slot) as usize * POWERS[WINDOW_LEN - 1 - slot])
        .sum()
}

fn longest_run(code: usize, value: u8) -> u8 {
    let (mut run, mut best) = (0, 0);
    for slot in 0..WINDOW_LEN {
        if digit(code, slot) == value {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

fn generate_pattern_tables() -> PatternTables {
    let mut reverse = [0u16; TERNARY_CODES];
    let mut pattern_class = [-1i64; TERNARY_CODES];
    let mut own_count = [0u8; TERNARY_CODES];
    let mut opp_count = [0u8; TERNARY_CODES];
    let mut empty_count = [0u8; TERNARY_CODES];
    let mut own_max_run = [0u8; TERNARY_CODES];
    let mut opp_max_run = [0u8; TERNARY_CODES];
    let mut status = [0u8; TERNARY_CODES];

    for code in 0..TERNARY_CODES {
        reverse[code] = reverse_code(code) as u16;
        for slot in 0..WINDOW_LEN {
            match digit(code, slot) {
                OWN => own_count[code] += 1,
                OPP => opp_count[code] += 1,
                _ => {}
            }
        }
        empty_count[code] = WINDOW_LEN as u8 - own_count[code] - opp_count[code];
        own_max_run[code] = longest_run(code, OWN);
        opp_max_run[code] = longest_run(code, OPP);
        status[code] =
            u8::from(own_count[code] > 0) * OWN_LIVE + u8::from(opp_count[code] > 0) * OPP_LIVE;
    }

    let mut next = 0i64;
    for code in 0..TERNARY_CODES {
        let representative = code.min(reverse[code] as usize);
        if representative == code {
            pattern_class[code] = next;
            next += 1;
        }
    }
    assert_eq!(next, EXPECTED_PATTERN_CLASSES);
    for code in 0..TERNARY_CODES {
        let representative = code.min(reverse[code] as usize);
        pattern_class[code] = pattern_class[representative];
    }

    let mut cell_window_class = [-1i64; CODE_SLOT_PAIRS];
    next = 0;
    for (pair, class) in cell_window_class.iter_mut().enumerate() {
        let code = pair / WINDOW_LEN;
        let slot = pair % WINDOW_LEN;
        let partner = reverse[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
        if pair <= partner {
            *class = next;
            next += 1;
        }
    }
    assert_eq!(next, EXPECTED_CELL_WINDOW_CLASSES);
    let representative_classes = cell_window_class;
    for (pair, class) in cell_window_class.iter_mut().enumerate() {
        let code = pair / WINDOW_LEN;
        let slot = pair % WINDOW_LEN;
        let partner = reverse[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
        *class = representative_classes[pair.min(partner)];
    }

    let mut post1_class = [-1i64; CODE_SLOT_PAIRS];
    next = 0;
    for (pair, class) in post1_class.iter_mut().enumerate() {
        let code = pair / WINDOW_LEN;
        let slot = pair % WINDOW_LEN;
        if digit(code, slot) != OWN {
            continue;
        }
        let partner = reverse[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
        if pair <= partner {
            *class = next;
            next += 1;
        }
    }
    assert_eq!(next, EXPECTED_POST1_CLASSES);
    let representative_classes = post1_class;
    for (pair, class) in post1_class.iter_mut().enumerate() {
        let code = pair / WINDOW_LEN;
        let slot = pair % WINDOW_LEN;
        if digit(code, slot) != OWN {
            continue;
        }
        let partner = reverse[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
        *class = representative_classes[pair.min(partner)];
    }

    PatternTables {
        reverse_code: reverse,
        pattern_class,
        cell_window_class,
        post1_class,
        own_count,
        opp_count,
        empty_count,
        own_max_run,
        opp_max_run,
        status,
    }
}

fn pattern_tables() -> &'static PatternTables {
    static TABLES: OnceLock<PatternTables> = OnceLock::new();
    TABLES.get_or_init(generate_pattern_tables)
}

#[derive(Debug)]
struct OrbitTable {
    grid: [i8; (2 * MAX_ORBIT_RADIUS + 1) * (2 * MAX_ORBIT_RADIUS + 1)],
}

fn hex_distance_components(q: i16, r: i16) -> usize {
    let (q, r) = (i32::from(q), i32::from(r));
    q.abs().max(r.abs()).max((q + r).abs()) as usize
}

fn rotate((q, r): (i16, i16)) -> (i16, i16) {
    (-r, q + r)
}

fn reflect((q, r): (i16, i16)) -> (i16, i16) {
    (r, q)
}

fn canonical_displacement(displacement: (i16, i16)) -> (i16, i16) {
    let mut best = displacement;
    for reflected in [false, true] {
        let mut image = if reflected {
            reflect(displacement)
        } else {
            displacement
        };
        for _ in 0..6 {
            best = best.min(image);
            image = rotate(image);
        }
    }
    best
}

fn generate_orbit_table() -> OrbitTable {
    let radius = MAX_ORBIT_RADIUS as i16;
    let mut representatives = BTreeSet::new();
    for dq in -radius..=radius {
        for dr in -radius..=radius {
            let distance = hex_distance_components(dq, dr);
            if (1..=MAX_ORBIT_RADIUS).contains(&distance) {
                let (cq, cr) = canonical_displacement((dq, dr));
                representatives.insert((distance, cq, cr));
            }
        }
    }
    assert_eq!(representatives.len(), 48);
    let canonical: Vec<_> = representatives
        .iter()
        .map(|&(_distance, q, r)| (q, r))
        .collect();
    let rank: FxHashMap<_, _> = canonical
        .iter()
        .copied()
        .enumerate()
        .map(|(id, representative)| (representative, id as i8))
        .collect();
    let side = 2 * MAX_ORBIT_RADIUS + 1;
    let mut grid = [-1i8; (2 * MAX_ORBIT_RADIUS + 1) * (2 * MAX_ORBIT_RADIUS + 1)];
    for dq in -radius..=radius {
        for dr in -radius..=radius {
            let d = hex_distance_components(dq, dr);
            if (1..=MAX_ORBIT_RADIUS).contains(&d) {
                let index = (dq + radius) as usize * side + (dr + radius) as usize;
                grid[index] = rank[&canonical_displacement((dq, dr))];
            }
        }
    }
    OrbitTable { grid }
}

fn orbit_table() -> &'static OrbitTable {
    static TABLE: OnceLock<OrbitTable> = OnceLock::new();
    TABLE.get_or_init(generate_orbit_table)
}

fn orbit_id_at_distance(dq: i16, dr: i16, distance: usize, d_max: usize) -> Result<i64, String> {
    if !(1..=d_max).contains(&distance) {
        return Err(format!(
            "displacement ({dq}, {dr}) has hex distance {distance}, outside the 1..{d_max} orbit table"
        ));
    }
    let radius = MAX_ORBIT_RADIUS as i16;
    let side = 2 * MAX_ORBIT_RADIUS + 1;
    let index = (dq + radius) as usize * side + (dr + radius) as usize;
    Ok(i64::from(orbit_table().grid[index]))
}

#[cfg(test)]
fn orbit_id(dq: i16, dr: i16, d_max: usize) -> Result<i64, String> {
    orbit_id_at_distance(dq, dr, hex_distance_components(dq, dr), d_max)
}

fn offset_coord(coord: engine::HexCoord, dq: i16, dr: i16, n: i16) -> engine::HexCoord {
    let q = i32::from(coord.q) + i32::from(dq) * i32::from(n);
    let r = i32::from(coord.r) + i32::from(dr) * i32::from(n);
    debug_assert!((i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&q));
    debug_assert!((i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&r));
    engine::HexCoord::new(q as i16, r as i16)
}

fn window_start(seed: engine::HexCoord, axis: usize, slot: usize) -> engine::HexCoord {
    let (dq, dr) = AXES[axis];
    offset_coord(seed, dq, dr, -(slot as i16))
}

fn window_cell(id: WindowId, slot: usize) -> engine::HexCoord {
    let (dq, dr) = AXES[id.axis as usize];
    offset_coord(engine::HexCoord::new(id.q, id.r), dq, dr, slot as i16)
}

fn relative_digit(pos: &engine::Position, mover: engine::Player, coord: engine::HexCoord) -> u8 {
    match pos.get(coord) {
        None => EMPTY,
        Some(owner) if owner == mover => OWN,
        Some(_) => OPP,
    }
}

fn window_code(pos: &engine::Position, mover: engine::Player, id: WindowId) -> usize {
    (0..WINDOW_LEN)
        .map(|slot| relative_digit(pos, mover, window_cell(id, slot)) as usize * POWERS[slot])
        .sum()
}

fn position_moves_remaining(pos: &engine::Position) -> u8 {
    match pos.phase() {
        engine::TurnPhase::FirstStone => 2,
        engine::TurnPhase::Opening | engine::TurnPhase::SecondStone => 1,
    }
}

fn fraction(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (count as f64 / total as f64) as f32
    }
}

struct WindowSet {
    ids: Vec<WindowId>,
    code: Vec<usize>,
    status: Vec<u8>,
    index: FxHashMap<WindowId, i32>,
}

fn enumerate_windows(
    pos: &engine::Position,
    mover: engine::Player,
    stones: &[(engine::HexCoord, engine::Player)],
    legal: &[engine::HexCoord],
    scope: WindowScope,
) -> Result<WindowSet, String> {
    let seed_count = stones.len() + usize::from(scope == WindowScope::ActionRelevant) * legal.len();
    let mut candidates = Vec::with_capacity(seed_count * POST_ACTION_ROWS);
    let seeds = stones.iter().map(|&(coord, _)| coord).chain(
        (scope == WindowScope::ActionRelevant)
            .then_some(legal.iter().copied())
            .into_iter()
            .flatten(),
    );
    for seed in seeds {
        for axis in 0..NUM_AXES {
            for slot in 0..WINDOW_LEN {
                let start = window_start(seed, axis, slot);
                candidates.push(WindowId {
                    axis: axis as u8,
                    q: start.q,
                    r: start.r,
                });
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();

    let tables = pattern_tables();
    let mut ids = Vec::with_capacity(candidates.len());
    let mut codes = Vec::with_capacity(candidates.len());
    let mut statuses = Vec::with_capacity(candidates.len());
    for id in candidates {
        let code = window_code(pos, mover, id);
        if matches!(code, ALL_OWN_CODE | ALL_OPP_CODE) {
            let colour = if code == ALL_OWN_CODE {
                "own"
            } else {
                "opponent"
            };
            return Err(format!(
                "window (axis={}, start=({}, {})) holds six {colour} stones: the position is terminal and the builder refuses it",
                id.axis, id.q, id.r
            ));
        }
        let status = tables.status[code];
        let keep = match scope {
            WindowScope::Live => matches!(status, OWN_LIVE | OPP_LIVE),
            WindowScope::Nonempty => status != EMPTY,
            WindowScope::ActionRelevant => true,
        };
        if keep {
            ids.push(id);
            codes.push(code);
            statuses.push(status);
        }
    }
    let index = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(row, id)| {
            i32::try_from(row)
                .map(|row| (id, row))
                .map_err(|_| format!("window row {row} exceeds the compact action index frame"))
        })
        .collect::<Result<_, _>>()?;
    Ok(WindowSet {
        ids,
        code: codes,
        status: statuses,
        index,
    })
}

struct CellSet {
    coords: Vec<engine::HexCoord>,
    index: FxHashMap<engine::HexCoord, i64>,
    occupancy: Vec<i64>,
    is_legal: Vec<i64>,
    is_occupied: Vec<i64>,
    nearest_bucket: Vec<i64>,
    legal_to_cell: Vec<i64>,
}

type IncidenceColumns = (Vec<i64>, Vec<i64>, Vec<bool>);
type RadiusColumns = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>);
type ActionColumns = (Vec<i64>, Vec<i64>, Vec<i64>);

#[derive(Clone, Copy, Default)]
struct RadiusEdge {
    source: u32,
    destination: u32,
    relation: u16,
    route: i8,
}

struct RadiusPlan {
    edges: Vec<RadiusEdge>,
    cell_count: usize,
}

impl RadiusPlan {
    fn rows(&self) -> usize {
        self.edges.len()
    }
}

struct ActionPlan {
    window_index: Vec<i32>,
    post1_class: Vec<u16>,
    pre_status: Vec<u8>,
    window_count: usize,
    legal_count: usize,
}

impl ActionPlan {
    fn rows(&self) -> usize {
        self.window_index.len()
    }
}

enum CleanupJob {
    Graphs(Vec<ActGraph>),
    Plans {
        radius: Vec<Option<RadiusPlan>>,
        action: Vec<ActionPlan>,
    },
}

// Large position-local vectors are dead as soon as their packed copies exist.
// Destroying hundreds of allocator-backed chunks on the caller serialized the
// next batch. Each calling worker owns a bounded queue holding at most its
// graph/plan pair, so two fitloop workers cannot grow an unbounded shared
// backlog. The process id prevents a sender inherited across POSIX fork from
// targeting a receiver thread that no longer exists in the child. The sender
// is cloned out of the RefCell before a potentially blocking send.
thread_local! {
    static CLEANUP_SENDER: RefCell<Option<(u32, std::sync::mpsc::SyncSender<CleanupJob>)>> =
        const { RefCell::new(None) };
}

fn cleanup_sender() -> Result<std::sync::mpsc::SyncSender<CleanupJob>, String> {
    let process_id = std::process::id();
    CLEANUP_SENDER.with(|slot| {
        let mut state = slot.borrow_mut();
        if state
            .as_ref()
            .is_none_or(|(owner_process, _)| *owner_process != process_id)
        {
            let (sender, receiver) = std::sync::mpsc::sync_channel(2);
            std::thread::Builder::new()
                .name("hexo-act-cleanup".to_owned())
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        match job {
                            CleanupJob::Graphs(graphs) => drop(graphs),
                            CleanupJob::Plans { radius, action } => drop((radius, action)),
                        }
                    }
                })
                .map_err(|error| format!("cannot start ACT buffer cleanup thread: {error}"))?;
            *state = Some((process_id, sender));
        }
        Ok(state
            .as_ref()
            .expect("cleanup sender initialized")
            .1
            .clone())
    })
}

fn defer_cleanup(job: CleanupJob) -> Result<(), String> {
    let sender = cleanup_sender()?;
    if sender.send(job).is_err() {
        CLEANUP_SENDER.with(|slot| *slot.borrow_mut() = None);
        return Err("ACT buffer cleanup thread stopped unexpectedly".to_owned());
    }
    Ok(())
}

fn relevant_cells(
    pos: &engine::Position,
    mover: engine::Player,
    stones: &[(engine::HexCoord, engine::Player)],
    legal: &[engine::HexCoord],
    windows: &WindowSet,
    scope: CellScope,
) -> Result<CellSet, String> {
    let mut coordinates = Vec::with_capacity(
        stones.len()
            + usize::from(scope != CellScope::OccupiedOnly) * legal.len()
            + usize::from(scope == CellScope::WindowAndLegal) * windows.ids.len() * WINDOW_LEN,
    );
    coordinates.extend(stones.iter().map(|&(coord, _)| coord));
    if scope != CellScope::OccupiedOnly {
        coordinates.extend(legal.iter().copied());
    }
    if scope == CellScope::WindowAndLegal {
        for &id in &windows.ids {
            coordinates.extend((0..WINDOW_LEN).map(|slot| window_cell(id, slot)));
        }
    }
    coordinates.sort_unstable();
    coordinates.dedup();
    let coords = coordinates;
    let index: FxHashMap<_, _> = coords
        .iter()
        .copied()
        .enumerate()
        .map(|(row, coord)| (coord, row as i64))
        .collect();

    let mut occupancy = vec![0i64; coords.len()];
    let mut is_occupied = vec![0i64; coords.len()];
    for &(coord, owner) in stones {
        let row = index[&coord] as usize;
        occupancy[row] = if owner == mover { 1 } else { 2 };
        is_occupied[row] = 1;
    }
    let mut is_legal = vec![0i64; coords.len()];
    let mut legal_to_cell = Vec::with_capacity(legal.len());
    for &coord in legal {
        let row = index.get(&coord).copied().unwrap_or(-1);
        if scope == CellScope::OccupiedOnly {
            debug_assert_eq!(row, -1);
            legal_to_cell.push(-1);
        } else {
            if row < 0 {
                return Err(format!(
                    "legal cell ({}, {}) is absent from the configured cell set",
                    coord.q, coord.r
                ));
            }
            if pos.get(coord).is_some() {
                return Err(format!(
                    "legal cell ({}, {}) holds a stone: a placement is legal only on an empty cell",
                    coord.q, coord.r
                ));
            }
            is_legal[row as usize] = 1;
            legal_to_cell.push(row);
        }
    }

    let mut nearest_bucket = vec![NEAREST_UNREACHED; coords.len()];
    let radius = engine::LEGAL_RADIUS as i16;
    for &(stone, _) in stones {
        for dq in -radius..=radius {
            for dr in -radius..=radius {
                let distance = hex_distance_components(dq, dr);
                if distance > engine::LEGAL_RADIUS as usize {
                    continue;
                }
                let coord = offset_coord(stone, dq, dr, 1);
                if let Some(&row) = index.get(&coord) {
                    nearest_bucket[row as usize] =
                        nearest_bucket[row as usize].min(distance as i64);
                }
            }
        }
    }

    Ok(CellSet {
        coords,
        index,
        occupancy,
        is_legal,
        is_occupied,
        nearest_bucket,
        legal_to_cell,
    })
}

fn incidence(
    windows: &WindowSet,
    cells: &CellSet,
    scope: CellScope,
) -> Result<IncidenceColumns, String> {
    let tables = pattern_tables();
    let rows = windows.ids.len() * WINDOW_LEN;
    let mut cell_index = Vec::with_capacity(rows);
    let mut class = Vec::with_capacity(rows);
    let mut mask = Vec::with_capacity(rows);
    for (window, (&id, &code)) in windows.ids.iter().zip(&windows.code).enumerate() {
        for slot in 0..WINDOW_LEN {
            let coord = window_cell(id, slot);
            let found = cells.index.get(&coord).copied().unwrap_or(-1);
            if scope == CellScope::WindowAndLegal && found < 0 {
                return Err(format!(
                    "window ({}, {}, {}) slot {slot} is absent from the window_and_legal cell set at window {window}",
                    id.axis, id.q, id.r
                ));
            }
            cell_index.push(found);
            mask.push(found >= 0);
            class.push(if found >= 0 {
                tables.cell_window_class[code * WINDOW_LEN + slot]
            } else {
                -1
            });
        }
    }
    Ok((cell_index, class, mask))
}

fn adjacency_edges(cells: &CellSet) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut edges = Vec::with_capacity(cells.coords.len() * 6);
    for (source, &coord) in cells.coords.iter().enumerate() {
        for (axis, &(dq, dr)) in AXES.iter().enumerate() {
            for sign in [1, -1] {
                let destination = offset_coord(coord, dq, dr, sign);
                if let Some(&destination) = cells.index.get(&destination) {
                    edges.push((destination, source as i64, axis as i64));
                }
            }
        }
    }
    edges.sort_unstable_by_key(|&(destination, source, relation)| (destination, source, relation));
    let mut source = Vec::with_capacity(edges.len());
    let mut destination = Vec::with_capacity(edges.len());
    let mut axis = Vec::with_capacity(edges.len());
    for (dst, src, relation) in edges {
        source.push(src);
        destination.push(dst);
        axis.push(relation);
    }
    (source, destination, axis)
}

fn on_axis(dq: i16, dr: i16) -> i64 {
    if dq == 0 && dr == 0 {
        -1
    } else if dr == 0 {
        0
    } else if dq == 0 {
        1
    } else if dq + dr == 0 {
        2
    } else {
        -1
    }
}

fn radius_relation(
    dq: i16,
    dr: i16,
    distance: usize,
    cfg: &ActBuilderConfig,
) -> Result<(i64, i64), String> {
    let route = on_axis(dq, dr);
    let relation = match cfg.d6_relation_mode {
        D6RelationMode::Orbit48 => orbit_id_at_distance(dq, dr, distance, cfg.d_max)?,
        D6RelationMode::CoarseDistanceAxis => {
            (2 * (distance.min(cfg.d_max) - 1)) as i64 + i64::from(route < 0)
        }
    };
    Ok((relation, route))
}

fn plan_radius_edges(
    stones: &[(engine::HexCoord, engine::Player)],
    coords: Vec<engine::HexCoord>,
    index: FxHashMap<engine::HexCoord, i64>,
    cfg: &ActBuilderConfig,
) -> Result<RadiusPlan, String> {
    let mut occupied: Vec<_> = stones
        .iter()
        .map(|&(coord, _)| (index[&coord], coord))
        .collect();
    occupied.sort_unstable_by_key(|&(source, _)| source);
    let radius = cfg.occupied_radius as i16;
    let diameter = 2 * cfg.occupied_radius + 1;
    let direct_work = occupied.len().saturating_mul(coords.len());
    let neighbor_work = occupied
        .len()
        .saturating_mul(diameter.saturating_mul(diameter));
    let maximum_edges = occupied
        .len()
        .saturating_mul(3 * cfg.occupied_radius * (cfg.occupied_radius + 1))
        .min(direct_work);
    let mut edges = Vec::with_capacity(maximum_edges);
    if direct_work <= neighbor_work {
        for (destination_index, &destination_coord) in coords.iter().enumerate() {
            let destination = u32::try_from(destination_index)
                .map_err(|_| format!("radius destination index {destination_index} exceeds u32"))?;
            for &(source_index, stone) in &occupied {
                let dq = destination_coord.q - stone.q;
                let dr = destination_coord.r - stone.r;
                let distance = hex_distance_components(dq, dr);
                if !(1..=cfg.occupied_radius).contains(&distance) {
                    continue;
                }
                let (relation, route) = radius_relation(dq, dr, distance, cfg)?;
                edges.push(RadiusEdge {
                    source: u32::try_from(source_index)
                        .map_err(|_| format!("radius source index {source_index} exceeds u32"))?,
                    destination,
                    relation: u16::try_from(relation)
                        .map_err(|_| format!("radius relation {relation} exceeds u16"))?,
                    route: i8::try_from(route)
                        .map_err(|_| format!("radius route {route} exceeds i8"))?,
                });
            }
        }
    } else {
        let mut counts = vec![0usize; coords.len()];
        for &(source_index, stone) in &occupied {
            let source = u32::try_from(source_index)
                .map_err(|_| format!("radius source index {source_index} exceeds u32"))?;
            for dq in -radius..=radius {
                for dr in -radius..=radius {
                    let distance = hex_distance_components(dq, dr);
                    if !(1..=cfg.occupied_radius).contains(&distance) {
                        continue;
                    }
                    let destination_coord = offset_coord(stone, dq, dr, 1);
                    if let Some(&destination_index) = index.get(&destination_coord) {
                        counts[destination_index as usize] += 1;
                        let (relation, route) = radius_relation(dq, dr, distance, cfg)?;
                        edges.push(RadiusEdge {
                            source,
                            destination: u32::try_from(destination_index).map_err(|_| {
                                format!("radius destination index {destination_index} exceeds u32")
                            })?,
                            relation: u16::try_from(relation)
                                .map_err(|_| format!("radius relation {relation} exceeds u16"))?,
                            route: i8::try_from(route)
                                .map_err(|_| format!("radius route {route} exceeds i8"))?,
                        });
                    }
                }
            }
        }
        let mut starts = Vec::with_capacity(counts.len() + 1);
        starts.push(0);
        let mut total = 0usize;
        for count in counts {
            total = total
                .checked_add(count)
                .ok_or_else(|| "radius edge count overflows usize".to_owned())?;
            starts.push(total);
        }
        let mut next = starts[..coords.len()].to_vec();
        let mut ordered = vec![RadiusEdge::default(); edges.len()];
        for edge in edges {
            let destination = edge.destination as usize;
            let row = next[destination];
            next[destination] += 1;
            ordered[row] = edge;
        }
        edges = ordered;
    }
    Ok(RadiusPlan {
        edges,
        cell_count: coords.len(),
    })
}

fn fill_radius_edges(
    plan: &RadiusPlan,
    cell_offset: i64,
    columns: (&mut [i64], &mut [i64], &mut [i64], &mut [i64]),
) -> Result<i64, String> {
    let (source, destination, relation, route) = columns;
    let rows = plan.rows();
    if [source.len(), destination.len(), relation.len(), route.len()]
        .into_iter()
        .any(|length| length != rows)
    {
        return Err("radius output slices disagree with the planned row count".to_owned());
    }
    let mut relation_bound = 0i64;
    for (row, edge) in plan.edges.iter().enumerate() {
        if edge.source as usize >= plan.cell_count || edge.destination as usize >= plan.cell_count {
            return Err(format!(
                "radius plan row {row} indexes ({}, {}) against {} cells",
                edge.source, edge.destination, plan.cell_count
            ));
        }
        source[row] = i64::from(edge.source)
            .checked_add(cell_offset)
            .ok_or_else(|| "radius source plus packed cell offset overflows i64".to_owned())?;
        destination[row] = i64::from(edge.destination)
            .checked_add(cell_offset)
            .ok_or_else(|| "radius destination plus packed cell offset overflows i64".to_owned())?;
        relation[row] = i64::from(edge.relation);
        route[row] = i64::from(edge.route);
        relation_bound = relation_bound.max(relation[row] + 1);
    }
    Ok(relation_bound)
}

fn materialize_radius_edges(plan: &RadiusPlan) -> Result<RadiusColumns, String> {
    let rows = plan.rows();
    let mut source = vec![0; rows];
    let mut destination = vec![0; rows];
    let mut relation = vec![0; rows];
    let mut route = vec![0; rows];
    let _ = fill_radius_edges(
        plan,
        0,
        (&mut source, &mut destination, &mut relation, &mut route),
    )?;
    Ok((source, destination, relation, route))
}

struct ActionTables {
    window_index: Vec<i32>,
    post1_class: Vec<u16>,
    pre_status: Vec<u8>,
    pre_code: Vec<u16>,
    post_code: Vec<u16>,
}

impl ActionTables {
    fn into_plan(self, window_count: usize, legal_count: usize) -> ActionPlan {
        ActionPlan {
            window_index: self.window_index,
            post1_class: self.post1_class,
            pre_status: self.pre_status,
            window_count,
            legal_count,
        }
    }
}

fn action_tables(
    pos: &engine::Position,
    mover: engine::Player,
    legal: &[engine::HexCoord],
    windows: &WindowSet,
    scope: WindowScope,
) -> Result<ActionTables, String> {
    let tables = pattern_tables();
    let rows = legal
        .len()
        .checked_mul(POST_ACTION_ROWS)
        .ok_or_else(|| "action row count overflows usize".to_owned())?;
    let mut window_index = Vec::with_capacity(rows);
    let mut post1_class = Vec::with_capacity(rows);
    let mut pre_status = Vec::with_capacity(rows);
    let mut pre_code = Vec::with_capacity(rows);
    let mut post_code = Vec::with_capacity(rows);

    for (action, &coord) in legal.iter().enumerate() {
        if pos.get(coord).is_some() {
            return Err(format!(
                "legal action {action} at ({}, {}) is occupied",
                coord.q, coord.r
            ));
        }
        for (axis, &(dq, dr)) in AXES.iter().enumerate() {
            let mut line = [0u8; 2 * WINDOW_LEN - 1];
            for (line_slot, value) in line.iter_mut().enumerate() {
                *value = relative_digit(
                    pos,
                    mover,
                    offset_coord(coord, dq, dr, line_slot as i16 - 5),
                );
            }
            if line[WINDOW_LEN - 1] != EMPTY {
                return Err(format!(
                    "legal action {action} at ({}, {}) is occupied by colour {}",
                    coord.q,
                    coord.r,
                    line[WINDOW_LEN - 1]
                ));
            }
            for (candidate_slot, &candidate_power) in POWERS.iter().enumerate() {
                let pre: usize = (0..WINDOW_LEN)
                    .map(|window_slot| {
                        let line_slot = window_slot + WINDOW_LEN - 1 - candidate_slot;
                        line[line_slot] as usize * POWERS[window_slot]
                    })
                    .sum();
                let post = pre + candidate_power;
                let start = window_start(coord, axis, candidate_slot);
                let id = WindowId {
                    axis: axis as u8,
                    q: start.q,
                    r: start.r,
                };
                let found = windows.index.get(&id).copied().unwrap_or(-1);
                let expected = match scope {
                    WindowScope::ActionRelevant => true,
                    WindowScope::Nonempty => pre != 0,
                    WindowScope::Live => {
                        matches!(tables.status[pre], OWN_LIVE | OPP_LIVE)
                    }
                };
                if (found >= 0) != expected {
                    return Err(format!(
                        "the {scope:?} window set disagrees at legal action {action}, axis {axis}, candidate slot {candidate_slot}, pre-placement code {pre}"
                    ));
                }
                let class = tables.post1_class[post * WINDOW_LEN + candidate_slot];
                if class < 0 {
                    return Err(format!(
                        "post-placement code {post} does not hold an own stone at candidate slot {candidate_slot}"
                    ));
                }
                window_index.push(found);
                post1_class.push(u16::try_from(class).map_err(|_| {
                    format!("post-placement class {class} exceeds the compact action frame")
                })?);
                pre_status.push(tables.status[pre]);
                pre_code.push(u16::try_from(pre).map_err(|_| {
                    format!("pre-placement code {pre} exceeds the compact action frame")
                })?);
                post_code.push(u16::try_from(post).map_err(|_| {
                    format!("post-placement code {post} exceeds the compact action frame")
                })?);
            }
        }
    }
    Ok(ActionTables {
        window_index,
        post1_class,
        pre_status,
        pre_code,
        post_code,
    })
}

fn validate_action_plan(plan: &ActionPlan) -> Result<usize, String> {
    let expected_rows = plan
        .legal_count
        .checked_mul(POST_ACTION_ROWS)
        .ok_or_else(|| "action row count overflows usize".to_owned())?;
    if plan.rows() != expected_rows
        || plan.post1_class.len() != expected_rows
        || plan.pre_status.len() != expected_rows
    {
        return Err(format!(
            "action plan for {} legal rows has {}, {}, and {} values; expected {expected_rows}",
            plan.legal_count,
            plan.rows(),
            plan.post1_class.len(),
            plan.pre_status.len()
        ));
    }
    Ok(expected_rows)
}

fn action_row_values(
    plan: &ActionPlan,
    row: usize,
    window_offset: i64,
) -> Result<(i64, i64, i64), String> {
    let local_window = i64::from(plan.window_index[row]);
    let window_index = if local_window == -1 {
        -1
    } else if local_window < 0 || local_window as usize >= plan.window_count {
        return Err(format!(
            "action plan row {row} indexes window {local_window} against {} windows",
            plan.window_count
        ));
    } else {
        local_window.checked_add(window_offset).ok_or_else(|| {
            "action window index plus packed window offset overflows i64".to_owned()
        })?
    };

    let class = i64::from(plan.post1_class[row]);
    if class >= EXPECTED_POST1_CLASSES {
        return Err(format!(
            "action plan row {row} has post1 class {class} outside 0..{EXPECTED_POST1_CLASSES}"
        ));
    }

    let status = i64::from(plan.pre_status[row]);
    if status > i64::from(MIXED) {
        return Err(format!(
            "action plan row {row} has pre-status {status} outside 0..={MIXED}"
        ));
    }
    Ok((window_index, class, status))
}

fn fill_action_rows(
    plan: &ActionPlan,
    window_offset: i64,
    columns: (&mut [i64], &mut [i64], &mut [i64]),
) -> Result<(), String> {
    let (window_index, post1_class, pre_status) = columns;
    let expected_rows = validate_action_plan(plan)?;
    if [window_index.len(), post1_class.len(), pre_status.len()]
        .into_iter()
        .any(|length| length != expected_rows)
    {
        return Err("action output slices disagree with the planned row count".to_owned());
    }

    for row in 0..expected_rows {
        (window_index[row], post1_class[row], pre_status[row]) =
            action_row_values(plan, row, window_offset)?;
    }
    Ok(())
}

fn materialize_action_rows(plan: &ActionPlan) -> Result<ActionColumns, String> {
    let rows = plan.rows();
    let mut window_index = vec![0; rows];
    let mut post1_class = vec![0; rows];
    let mut pre_status = vec![0; rows];
    fill_action_rows(
        plan,
        0,
        (&mut window_index, &mut post1_class, &mut pre_status),
    )?;
    Ok((window_index, post1_class, pre_status))
}

fn tactical_features(actions: &ActionTables, n_legal: usize) -> Vec<f32> {
    let tables = pattern_tables();
    let mut five_windows = FxHashSet::default();
    let mut four_windows = FxHashSet::default();
    for row in 0..actions.pre_code.len() {
        let pre = usize::from(actions.pre_code[row]);
        if tables.status[pre] == OPP_LIVE {
            match tables.opp_count[pre] {
                5 => {
                    five_windows.insert(actions.window_index[row]);
                }
                4 => {
                    four_windows.insert(actions.window_index[row]);
                }
                _ => {}
            }
        }
    }
    debug_assert!(!five_windows.contains(&-1));
    debug_assert!(!four_windows.contains(&-1));
    let five_remaining = five_windows.len();
    let four_remaining = four_windows.len();
    let five_global = fraction(five_remaining.min(GLOBAL_THREAT_CAP), GLOBAL_THREAT_CAP);
    let four_global = fraction(four_remaining.min(GLOBAL_THREAT_CAP), GLOBAL_THREAT_CAP);

    let mut features = Vec::with_capacity(n_legal * TACTICAL_FEATURES);
    for action in 0..n_legal {
        let rows = action * POST_ACTION_ROWS..(action + 1) * POST_ACTION_ROWS;
        let mut immediate_win = false;
        let mut max_own_after = 0usize;
        let mut max_opp_before = 0usize;
        let mut own_five_after = 0usize;
        let mut own_four_after = 0usize;
        let mut opp_five_hit = 0usize;
        let mut opp_four_hit = 0usize;
        let mut mixed_created = 0usize;
        let mut nonempty_pre = 0usize;
        for row in rows {
            let pre = usize::from(actions.pre_code[row]);
            let post = usize::from(actions.post_code[row]);
            let own_after = tables.own_count[post] as usize;
            let opp_before = tables.opp_count[pre] as usize;
            immediate_win |= post == ALL_OWN_CODE;
            max_own_after = max_own_after.max(own_after);
            max_opp_before = max_opp_before.max(opp_before);
            if tables.status[post] == OWN_LIVE {
                own_five_after += usize::from(own_after == 5);
                own_four_after += usize::from(own_after == 4);
            }
            if tables.status[pre] == OPP_LIVE {
                opp_five_hit += usize::from(opp_before == 5);
                opp_four_hit += usize::from(opp_before == 4);
                mixed_created += 1;
            }
            nonempty_pre += usize::from(pre != 0);
        }
        features.extend([
            u8::from(immediate_win) as f32,
            fraction(max_own_after, WINDOW_LEN),
            fraction(max_opp_before, WINDOW_LEN),
            fraction(own_five_after, POST_ACTION_ROWS),
            fraction(own_four_after, POST_ACTION_ROWS),
            fraction(opp_five_hit, POST_ACTION_ROWS),
            fraction(opp_four_hit, POST_ACTION_ROWS),
            five_global,
            four_global,
            u8::from(five_remaining > 0 && opp_five_hit == five_remaining) as f32,
            fraction(mixed_created, POST_ACTION_ROWS),
            fraction(nonempty_pre, POST_ACTION_ROWS),
        ]);
    }
    features
}

fn capped_aux_count(count: usize) -> i64 {
    count.min(AUX_COUNT_CAP) as i64
}

fn auxiliary_labels(
    actions: &ActionTables,
    legal: &[engine::HexCoord],
) -> Result<ActAuxLabels, String> {
    let expected_rows = legal.len() * POST_ACTION_ROWS;
    if actions.pre_code.len() != expected_rows || actions.post_code.len() != expected_rows {
        return Err(format!(
            "{} legal actions need {expected_rows} pre/post rows, got {} and {}",
            legal.len(),
            actions.pre_code.len(),
            actions.post_code.len()
        ));
    }

    let tables = pattern_tables();
    let mut win_now = Vec::with_capacity(legal.len());
    let mut own_max_occupancy = Vec::with_capacity(legal.len());
    let mut opponent_threats_hit = Vec::with_capacity(legal.len());
    let mut own_five_windows_after = Vec::with_capacity(legal.len());
    let mut winning_coordinates = FxHashSet::default();

    for (action, &coord) in legal.iter().enumerate() {
        let rows = action * POST_ACTION_ROWS..(action + 1) * POST_ACTION_ROWS;
        let mut wins = false;
        let mut own_max = 0usize;
        let mut threats_hit = 0usize;
        let mut own_fives = 0usize;
        for row in rows {
            let pre = usize::from(actions.pre_code[row]);
            let post = usize::from(actions.post_code[row]);
            let own_before = tables.own_count[pre] as usize;
            let opponent_before = tables.opp_count[pre] as usize;
            let own_after = tables.own_count[post] as usize;
            let opponent_after = tables.opp_count[post] as usize;
            wins |= own_after == WINDOW_LEN;
            own_max = own_max.max(own_after);
            threats_hit += usize::from(own_before == 0 && matches!(opponent_before, 4 | 5));
            own_fives += usize::from(own_after == 5 && opponent_after == 0);
        }
        if wins {
            winning_coordinates.insert((coord.q, coord.r));
        }
        win_now.push(i64::from(wins));
        own_max_occupancy.push(own_max as i64);
        opponent_threats_hit.push(capped_aux_count(threats_hit));
        own_five_windows_after.push(capped_aux_count(own_fives));
    }

    let always_winning = winning_coordinates.len();
    let mut winning_partner_exists = Vec::with_capacity(legal.len());
    let mut winning_partner_count = Vec::with_capacity(legal.len());
    for (action, &coord) in legal.iter().enumerate() {
        if win_now[action] != 0 {
            winning_partner_exists.push(0);
            winning_partner_count.push(0);
            continue;
        }

        let mut fresh_partners = FxHashSet::default();
        for (axis, &(dq, dr)) in AXES.iter().enumerate() {
            for candidate_slot in 0..WINDOW_LEN {
                let row = action * POST_ACTION_ROWS + axis * WINDOW_LEN + candidate_slot;
                let post = usize::from(actions.post_code[row]);
                if tables.own_count[post] as usize != 5 || tables.opp_count[post] != 0 {
                    continue;
                }
                let empty_slots: Vec<_> = (0..WINDOW_LEN)
                    .filter(|&slot| digit(post, slot) == EMPTY)
                    .collect();
                if empty_slots.len() != 1 {
                    return Err(format!(
                        "post-placement code {post} holds five own stones and no opponent stone but has {} empty slots",
                        empty_slots.len()
                    ));
                }
                let steps = empty_slots[0] as i16 - candidate_slot as i16;
                let partner = offset_coord(coord, dq, dr, steps);
                let key = (partner.q, partner.r);
                if !winning_coordinates.contains(&key) {
                    fresh_partners.insert(key);
                }
            }
        }
        let partners = always_winning + fresh_partners.len();
        winning_partner_exists.push(i64::from(partners > 0));
        winning_partner_count.push(capped_aux_count(partners));
    }

    Ok(ActAuxLabels {
        win_now,
        own_max_occupancy,
        opponent_threats_hit,
        own_five_windows_after,
        winning_partner_exists,
        winning_partner_count,
    })
}

/// Compute section 24.1's action labels from the authoritative engine position.
pub fn build_aux_labels(
    pos: &engine::Position,
    cfg: &ActBuilderConfig,
) -> Result<ActAuxLabels, String> {
    cfg.validate()?;
    if pos.is_terminal() {
        return Err("terminal position: the auxiliary-label builder refuses it".into());
    }
    let mover = pos.current_player();
    let stones: Vec<_> = pos.stones().collect();
    let legal: Vec<_> = pos.legal_actions().map(|action| action.coord()).collect();
    if legal.is_empty() {
        return Err("terminal position: the auxiliary-label builder refuses it".into());
    }
    let windows = enumerate_windows(pos, mover, &stones, &legal, cfg.window_scope)?;
    let actions = action_tables(pos, mover, &legal, &windows, cfg.window_scope)?;
    auxiliary_labels(&actions, &legal)
}

fn build_plan(
    pos: &engine::Position,
    cfg: &ActBuilderConfig,
) -> Result<(ActGraph, Option<RadiusPlan>, ActionPlan), String> {
    if pos.is_terminal() {
        return Err("terminal position: the builder refuses it".into());
    }
    let mover = pos.current_player();
    let moves_remaining = position_moves_remaining(pos);
    let stones: Vec<_> = pos.stones().collect();
    let legal: Vec<_> = pos.legal_actions().map(|action| action.coord()).collect();
    if legal.is_empty() {
        return Err("terminal position: the builder refuses it".into());
    }

    let windows = enumerate_windows(pos, mover, &stones, &legal, cfg.window_scope)?;
    let cells = relevant_cells(pos, mover, &stones, &legal, &windows, cfg.cell_scope)?;
    let (window_cell_index, window_incidence_class, window_incidence_mask) =
        incidence(&windows, &cells, cfg.cell_scope)?;
    let (adjacency_src, adjacency_dst, adjacency_axis) = if cfg.use_cell_adjacency {
        adjacency_edges(&cells)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let actions = action_tables(pos, mover, &legal, &windows, cfg.window_scope)?;

    let tables = pattern_tables();
    let mut window_id = Vec::with_capacity(windows.ids.len() * 3);
    let mut window_pattern_class = Vec::with_capacity(windows.ids.len());
    let mut window_axis = Vec::with_capacity(windows.ids.len());
    let mut window_numeric =
        Vec::with_capacity(windows.ids.len() * usize::from(cfg.use_window_numeric_features) * 5);
    for (&id, &code) in windows.ids.iter().zip(&windows.code) {
        window_id.extend([i64::from(id.axis), i64::from(id.q), i64::from(id.r)]);
        let representative = code.min(tables.reverse_code[code] as usize);
        window_pattern_class.push(tables.pattern_class[representative]);
        window_axis.push(i64::from(id.axis));
        if cfg.use_window_numeric_features {
            window_numeric.extend([
                fraction(tables.own_count[code] as usize, WINDOW_LEN),
                fraction(tables.opp_count[code] as usize, WINDOW_LEN),
                fraction(tables.empty_count[code] as usize, WINDOW_LEN),
                fraction(tables.own_max_run[code] as usize, WINDOW_LEN),
                fraction(tables.opp_max_run[code] as usize, WINDOW_LEN),
            ]);
        }
    }

    let action_tactical_numeric = if cfg.use_action_tactical_features {
        tactical_features(&actions, legal.len())
    } else {
        Vec::new()
    };
    let global_numeric = if cfg.use_global_numeric_features {
        let own_stones = stones.iter().filter(|&&(_, owner)| owner == mover).count();
        let own_live = windows
            .status
            .iter()
            .filter(|&&status| status == OWN_LIVE)
            .count();
        let opp_live = windows
            .status
            .iter()
            .filter(|&&status| status == OPP_LIVE)
            .count();
        let mixed = windows
            .status
            .iter()
            .filter(|&&status| status == MIXED)
            .count();
        vec![
            (stones.len() as f64).ln_1p() as f32,
            fraction(own_stones, stones.len()),
            fraction(stones.len() - own_stones, stones.len()),
            (legal.len() as f64).ln_1p() as f32,
            (windows.ids.len() as f64).ln_1p() as f32,
            fraction(own_live, windows.ids.len()),
            fraction(opp_live, windows.ids.len()),
            fraction(mixed, windows.ids.len()),
        ]
    } else {
        Vec::new()
    };
    let phase_id = if moves_remaining == 2 {
        1
    } else if stones.is_empty() {
        0
    } else {
        2
    };
    let action_plan = actions.into_plan(windows.ids.len(), legal.len());
    let mut cell_qr = Vec::with_capacity(cells.coords.len() * 2);
    for coord in &cells.coords {
        cell_qr.extend([i64::from(coord.q), i64::from(coord.r)]);
    }
    let CellSet {
        coords,
        index,
        occupancy,
        is_legal,
        is_occupied,
        nearest_bucket,
        legal_to_cell,
    } = cells;
    let radius_plan = if cfg.use_occupied_radius_edges {
        Some(plan_radius_edges(&stones, coords, index, cfg)?)
    } else {
        None
    };

    Ok((
        ActGraph {
            cell_qr,
            cell_occupancy: occupancy,
            cell_is_legal: is_legal,
            cell_is_occupied: is_occupied,
            cell_nearest_bucket: nearest_bucket,
            legal_to_cell_index: legal_to_cell,
            window_id,
            window_pattern_class,
            window_status: windows
                .status
                .iter()
                .map(|&value| i64::from(value))
                .collect(),
            window_axis,
            window_numeric,
            window_numeric_width: cfg.window_numeric_width(),
            window_cell_index,
            window_incidence_class,
            window_incidence_mask,
            adjacency_src,
            adjacency_dst,
            adjacency_axis,
            radius_src: Vec::new(),
            radius_dst: Vec::new(),
            radius_orbit: Vec::new(),
            radius_axis_or_neg1: Vec::new(),
            action_window_index: Vec::new(),
            action_post1_class: Vec::new(),
            action_pre_status: Vec::new(),
            action_tactical_numeric,
            action_tactical_numeric_width: cfg.tactical_width(),
            global_numeric,
            global_numeric_width: cfg.global_numeric_width(),
            moves_remaining,
            phase_id,
        },
        radius_plan,
        action_plan,
    ))
}

fn build_validated(pos: &engine::Position, cfg: &ActBuilderConfig) -> Result<ActGraph, String> {
    let (mut graph, radius_plan, action_plan) = build_plan(pos, cfg)?;
    (
        graph.action_window_index,
        graph.action_post1_class,
        graph.action_pre_status,
    ) = materialize_action_rows(&action_plan)?;
    if let Some(plan) = radius_plan {
        (
            graph.radius_src,
            graph.radius_dst,
            graph.radius_orbit,
            graph.radius_axis_or_neg1,
        ) = materialize_radius_edges(&plan)?;
    }
    Ok(graph)
}

/// Build one live engine position into a position-local MantisNet-ACT graph.
pub fn build(pos: &engine::Position, cfg: &ActBuilderConfig) -> Result<ActGraph, String> {
    cfg.validate()?;
    build_validated(pos, cfg)
}

fn family_offsets(
    family: &str,
    counts: impl IntoIterator<Item = usize>,
) -> Result<Vec<i64>, String> {
    let mut total = 0usize;
    let mut offsets = vec![0];
    for count in counts {
        total = total
            .checked_add(count)
            .ok_or_else(|| format!("{family} row count overflows usize"))?;
        offsets
            .push(i64::try_from(total).map_err(|_| {
                format!("{family} row count {total} exceeds the int64 index frame")
            })?);
    }
    Ok(offsets)
}

fn concatenate_copy<T, F>(graphs: &[ActGraph], expected: usize, field: F) -> Vec<T>
where
    T: Copy,
    F: for<'a> Fn(&'a ActGraph) -> &'a [T],
{
    let mut output = Vec::with_capacity(expected);
    for graph in graphs {
        output.extend_from_slice(field(graph));
    }
    debug_assert_eq!(output.len(), expected);
    output
}

fn concatenate_shifted<F>(
    graphs: &[ActGraph],
    expected: usize,
    target_offsets: &[i64],
    sentinel: bool,
    field: &'static str,
    select: F,
) -> Result<Vec<i64>, String>
where
    F: for<'a> Fn(&'a ActGraph) -> &'a [i64] + Sync,
{
    let mut output = Vec::with_capacity(expected);
    for (position, graph) in graphs.iter().enumerate() {
        let offset = target_offsets[position];
        let target_count = target_offsets[position + 1] - offset;
        for &value in select(graph) {
            if sentinel && value == -1 {
                output.push(-1);
            } else if value < 0 {
                return Err(format!(
                    "{field} contains negative index {value}{}",
                    if sentinel { " other than -1" } else { "" }
                ));
            } else if value >= target_count {
                return Err(format!(
                    "{field} contains local index {value} outside its target family's 0..{target_count} range"
                ));
            } else {
                output.push(value.checked_add(offset).ok_or_else(|| {
                    format!("{field} index {value} plus offset {offset} overflows i64")
                })?);
            }
        }
    }
    debug_assert_eq!(output.len(), expected);
    Ok(output)
}

/// Concatenate position-local graphs into the model's packed batch frame.
///
/// Graph order and every position's section-seven row order are preserved.
/// Index fields are shifted only here, against the target family's CSR
/// offset; consequently an emitted index cannot cross a position boundary.
pub fn collate(graphs: Vec<ActGraph>) -> Result<PackedActBatch, String> {
    collate_inner(graphs, false)
}

fn collate_inner(
    graphs: Vec<ActGraph>,
    action_rows_are_planned: bool,
) -> Result<PackedActBatch, String> {
    let Some(first) = graphs.first() else {
        return Err("empty batch: collate needs at least one position".into());
    };
    let window_numeric_width = first.window_numeric_width;
    let action_tactical_numeric_width = first.action_tactical_numeric_width;
    let global_numeric_width = first.global_numeric_width;
    for (index, graph) in graphs.iter().enumerate() {
        if graph.window_numeric_width != window_numeric_width {
            return Err(format!(
                "window_numeric has inconsistent feature widths across positions: position 0 has {window_numeric_width}, position {index} has {}",
                graph.window_numeric_width
            ));
        }
        if graph.action_tactical_numeric_width != action_tactical_numeric_width {
            return Err(format!(
                "action_tactical_numeric has inconsistent feature widths across positions: position 0 has {action_tactical_numeric_width}, position {index} has {}",
                graph.action_tactical_numeric_width
            ));
        }
        if graph.global_numeric_width != global_numeric_width {
            return Err(format!(
                "global_numeric has inconsistent feature widths across positions: position 0 has {global_numeric_width}, position {index} has {}",
                graph.global_numeric_width
            ));
        }
        let cells = graph.n_cells();
        let windows = graph.n_windows();
        let legal = graph.n_legal();
        let adjacency = graph.adjacency_src.len();
        let radius = graph.radius_src.len();
        let action_rows = if action_rows_are_planned {
            0
        } else {
            legal * POST_ACTION_ROWS
        };
        for (field, actual, expected) in [
            ("cell_qr", graph.cell_qr.len(), cells * 2),
            ("cell_is_legal", graph.cell_is_legal.len(), cells),
            ("cell_is_occupied", graph.cell_is_occupied.len(), cells),
            (
                "cell_nearest_bucket",
                graph.cell_nearest_bucket.len(),
                cells,
            ),
            ("window_id", graph.window_id.len(), windows * 3),
            ("window_status", graph.window_status.len(), windows),
            ("window_axis", graph.window_axis.len(), windows),
            (
                "window_numeric",
                graph.window_numeric.len(),
                windows * graph.window_numeric_width,
            ),
            (
                "window_cell_index",
                graph.window_cell_index.len(),
                windows * WINDOW_LEN,
            ),
            (
                "window_incidence_class",
                graph.window_incidence_class.len(),
                windows * WINDOW_LEN,
            ),
            (
                "window_incidence_mask",
                graph.window_incidence_mask.len(),
                windows * WINDOW_LEN,
            ),
            ("adjacency_dst", graph.adjacency_dst.len(), adjacency),
            ("adjacency_axis", graph.adjacency_axis.len(), adjacency),
            ("radius_dst", graph.radius_dst.len(), radius),
            ("radius_orbit", graph.radius_orbit.len(), radius),
            (
                "radius_axis_or_neg1",
                graph.radius_axis_or_neg1.len(),
                radius,
            ),
            (
                "action_window_index",
                graph.action_window_index.len(),
                action_rows,
            ),
            (
                "action_post1_class",
                graph.action_post1_class.len(),
                action_rows,
            ),
            (
                "action_pre_status",
                graph.action_pre_status.len(),
                action_rows,
            ),
            (
                "action_tactical_numeric",
                graph.action_tactical_numeric.len(),
                legal * graph.action_tactical_numeric_width,
            ),
            (
                "global_numeric",
                graph.global_numeric.len(),
                graph.global_numeric_width,
            ),
        ] {
            if actual != expected {
                return Err(format!(
                    "position {index} {field} has {actual} values, expected {expected}"
                ));
            }
        }
    }

    let position_count = graphs.len();
    let cell_offsets = family_offsets("cell", graphs.iter().map(ActGraph::n_cells))?;
    let window_offsets = family_offsets("window", graphs.iter().map(ActGraph::n_windows))?;
    let legal_offsets = family_offsets("legal", graphs.iter().map(ActGraph::n_legal))?;
    let adjacency_offsets = family_offsets(
        "adjacency",
        graphs.iter().map(|graph| graph.adjacency_src.len()),
    )?;
    let radius_offsets =
        family_offsets("radius", graphs.iter().map(|graph| graph.radius_src.len()))?;
    let cells = *cell_offsets.last().expect("offsets have a leading zero") as usize;
    let windows = *window_offsets.last().expect("offsets have a leading zero") as usize;
    let legal = *legal_offsets.last().expect("offsets have a leading zero") as usize;
    let adjacency = *adjacency_offsets
        .last()
        .expect("offsets have a leading zero") as usize;
    let radius = *radius_offsets.last().expect("offsets have a leading zero") as usize;

    let legal_to_cell_index = concatenate_shifted(
        &graphs,
        legal,
        &cell_offsets,
        true,
        "legal_to_cell_index",
        |graph| &graph.legal_to_cell_index,
    )?;
    let window_cell_index = concatenate_shifted(
        &graphs,
        windows * WINDOW_LEN,
        &cell_offsets,
        true,
        "window_cell_index",
        |graph| &graph.window_cell_index,
    )?;
    let adjacency_src = concatenate_shifted(
        &graphs,
        adjacency,
        &cell_offsets,
        false,
        "adjacency_src",
        |graph| &graph.adjacency_src,
    )?;
    let adjacency_dst = concatenate_shifted(
        &graphs,
        adjacency,
        &cell_offsets,
        false,
        "adjacency_dst",
        |graph| &graph.adjacency_dst,
    )?;
    let radius_src = concatenate_shifted(
        &graphs,
        radius,
        &cell_offsets,
        false,
        "radius_src",
        |graph| &graph.radius_src,
    )?;
    let radius_dst = concatenate_shifted(
        &graphs,
        radius,
        &cell_offsets,
        false,
        "radius_dst",
        |graph| &graph.radius_dst,
    )?;
    let action_window_index = concatenate_shifted(
        &graphs,
        if action_rows_are_planned {
            0
        } else {
            legal * POST_ACTION_ROWS
        },
        &window_offsets,
        true,
        "action_window_index",
        |graph| &graph.action_window_index,
    )?;

    let mut packed = PackedActBatch {
        position_count,
        cell_offsets,
        window_offsets,
        legal_offsets,
        adjacency_offsets,
        radius_offsets,
        cell_occupancy: concatenate_copy(&graphs, cells, |graph| &graph.cell_occupancy),
        cell_is_legal: concatenate_copy(&graphs, cells, |graph| &graph.cell_is_legal),
        cell_nearest_bucket: concatenate_copy(&graphs, cells, |graph| &graph.cell_nearest_bucket),
        legal_to_cell_index,
        window_id: concatenate_copy(&graphs, windows * 3, |graph| &graph.window_id),
        window_pattern_class: concatenate_copy(&graphs, windows, |graph| {
            &graph.window_pattern_class
        }),
        window_status: concatenate_copy(&graphs, windows, |graph| &graph.window_status),
        window_axis: concatenate_copy(&graphs, windows, |graph| &graph.window_axis),
        window_numeric: concatenate_copy(&graphs, windows * window_numeric_width, |graph| {
            &graph.window_numeric
        }),
        window_numeric_width,
        window_cell_index,
        window_incidence_class: concatenate_copy(&graphs, windows * WINDOW_LEN, |graph| {
            &graph.window_incidence_class
        }),
        window_incidence_mask: concatenate_copy(&graphs, windows * WINDOW_LEN, |graph| {
            &graph.window_incidence_mask
        }),
        adjacency_src,
        adjacency_dst,
        adjacency_axis: concatenate_copy(&graphs, adjacency, |graph| &graph.adjacency_axis),
        radius_src,
        radius_dst,
        radius_orbit: concatenate_copy(&graphs, radius, |graph| &graph.radius_orbit),
        radius_axis_or_neg1: concatenate_copy(&graphs, radius, |graph| &graph.radius_axis_or_neg1),
        action_window_index,
        action_post1_class: concatenate_copy(
            &graphs,
            if action_rows_are_planned {
                0
            } else {
                legal * POST_ACTION_ROWS
            },
            |graph| &graph.action_post1_class,
        ),
        action_pre_status: concatenate_copy(
            &graphs,
            if action_rows_are_planned {
                0
            } else {
                legal * POST_ACTION_ROWS
            },
            |graph| &graph.action_pre_status,
        ),
        action_tactical_numeric: concatenate_copy(
            &graphs,
            legal * action_tactical_numeric_width,
            |graph| &graph.action_tactical_numeric,
        ),
        action_tactical_numeric_width,
        phase_id: graphs
            .iter()
            .map(|graph| i64::from(graph.phase_id))
            .collect(),
        moves_remaining: graphs
            .iter()
            .map(|graph| i64::from(graph.moves_remaining))
            .collect(),
        global_numeric: concatenate_copy(&graphs, position_count * global_numeric_width, |graph| {
            &graph.global_numeric
        }),
        global_numeric_width,
        radius_orbit_bound: 0,
    };
    defer_cleanup(CleanupJob::Graphs(graphs))?;
    packed.radius_orbit_bound = packed
        .radius_orbit
        .par_iter()
        .copied()
        .max()
        .map_or(0, |maximum| maximum + 1);
    Ok(packed)
}

fn expand_plans(
    packed: &mut PackedActBatch,
    radius_plans: &[Option<RadiusPlan>],
    action_plans: &[ActionPlan],
) -> Result<(), String> {
    if radius_plans.len() != packed.position_count || action_plans.len() != packed.position_count {
        return Err(format!(
            "{} radius plans and {} action plans against {} packed positions",
            radius_plans.len(),
            action_plans.len(),
            packed.position_count
        ));
    }
    let action_rows = packed
        .legal_to_cell_index
        .len()
        .checked_mul(POST_ACTION_ROWS)
        .ok_or_else(|| "packed action row count overflows usize".to_owned())?;
    let radius_offsets = family_offsets(
        "radius",
        radius_plans
            .iter()
            .map(|plan| plan.as_ref().map_or(0, RadiusPlan::rows)),
    )?;
    let radius_rows = *radius_offsets
        .last()
        .expect("radius offsets have a leading zero") as usize;

    let mut window_index = vec![0; action_rows];
    let mut post1_class = vec![0; action_rows];
    let mut pre_status = vec![0; action_rows];
    let mut source = vec![0; radius_rows];
    let mut destination = vec![0; radius_rows];
    let mut relation = vec![0; radius_rows];
    let mut route = vec![0; radius_rows];
    let error = Mutex::new(None::<(usize, u8, String)>);
    let relation_bound = std::sync::atomic::AtomicI64::new(0);
    rayon::scope(|scope| {
        let mut window_tail = window_index.as_mut_slice();
        let mut class_tail = post1_class.as_mut_slice();
        let mut status_tail = pre_status.as_mut_slice();
        let mut source_tail = source.as_mut_slice();
        let mut destination_tail = destination.as_mut_slice();
        let mut relation_tail = relation.as_mut_slice();
        let mut route_tail = route.as_mut_slice();
        for position in 0..packed.position_count {
            let action_plan = &action_plans[position];
            let legal_count =
                (packed.legal_offsets[position + 1] - packed.legal_offsets[position]) as usize;
            let action_count = legal_count * POST_ACTION_ROWS;
            let (window_rows, window_rest) = window_tail.split_at_mut(action_count);
            let (class_rows, class_rest) = class_tail.split_at_mut(action_count);
            let (status_rows, status_rest) = status_tail.split_at_mut(action_count);
            let window_offset = packed.window_offsets[position];
            let radius_count = (radius_offsets[position + 1] - radius_offsets[position]) as usize;
            let (source_rows, source_rest) = source_tail.split_at_mut(radius_count);
            let (destination_rows, destination_rest) = destination_tail.split_at_mut(radius_count);
            let (relation_rows, relation_rest) = relation_tail.split_at_mut(radius_count);
            let (route_rows, route_rest) = route_tail.split_at_mut(radius_count);
            let radius_plan = radius_plans[position].as_ref();
            if radius_plan.is_none() {
                debug_assert_eq!(radius_count, 0);
            }
            let cell_offset = packed.cell_offsets[position];
            let plan_error = &error;
            let relation_bound = &relation_bound;
            scope.spawn(move |_| {
                let failure = match fill_action_rows(
                    action_plan,
                    window_offset,
                    (window_rows, class_rows, status_rows),
                ) {
                    Err(message) => Some((0, message)),
                    Ok(()) => radius_plan.and_then(|radius_plan| {
                        match fill_radius_edges(
                            radius_plan,
                            cell_offset,
                            (source_rows, destination_rows, relation_rows, route_rows),
                        ) {
                            Ok(bound) => {
                                relation_bound
                                    .fetch_max(bound, std::sync::atomic::Ordering::Relaxed);
                                None
                            }
                            Err(message) => Some((1, message)),
                        }
                    }),
                };
                if let Some((kind, message)) = failure {
                    let mut first = plan_error.lock().expect("plan error mutex poisoned");
                    if first
                        .as_ref()
                        .is_none_or(|current| (position, kind) < (current.0, current.1))
                    {
                        *first = Some((position, kind, message));
                    }
                }
            });
            window_tail = window_rest;
            class_tail = class_rest;
            status_tail = status_rest;
            source_tail = source_rest;
            destination_tail = destination_rest;
            relation_tail = relation_rest;
            route_tail = route_rest;
        }
    });
    if let Some((position, _, message)) = error.into_inner().expect("plan error mutex poisoned") {
        return Err(format!("position {position}: {message}"));
    }

    packed.action_window_index = window_index;
    packed.action_post1_class = post1_class;
    packed.action_pre_status = pre_status;
    packed.radius_offsets = radius_offsets;
    packed.radius_src = source;
    packed.radius_dst = destination;
    packed.radius_orbit = relation;
    packed.radius_axis_or_neg1 = route;
    packed.radius_orbit_bound = relation_bound.load(std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

fn collate_plans(
    planned: Vec<(ActGraph, Option<RadiusPlan>, ActionPlan)>,
) -> Result<PackedActBatch, String> {
    for (position, (graph, radius_plan, action_plan)) in planned.iter().enumerate() {
        if let Some(plan) = radius_plan
            && plan.cell_count != graph.n_cells()
        {
            return Err(format!(
                "position {position} radius plan has {} cells against graph's {}",
                plan.cell_count,
                graph.n_cells()
            ));
        }
        if action_plan.window_count != graph.n_windows()
            || action_plan.legal_count != graph.n_legal()
        {
            return Err(format!(
                "position {position} action plan has {} windows and {} legal rows against graph's {} and {}",
                action_plan.window_count,
                action_plan.legal_count,
                graph.n_windows(),
                graph.n_legal()
            ));
        }
    }
    let mut graphs = Vec::with_capacity(planned.len());
    let mut radius_plans = Vec::with_capacity(planned.len());
    let mut action_plans = Vec::with_capacity(planned.len());
    for (graph, radius_plan, action_plan) in planned {
        graphs.push(graph);
        radius_plans.push(radius_plan);
        action_plans.push(action_plan);
    }
    let mut packed = collate_inner(graphs, true)?;
    expand_plans(&mut packed, &radius_plans, &action_plans)?;
    defer_cleanup(CleanupJob::Plans {
        radius: radius_plans,
        action: action_plans,
    })?;
    Ok(packed)
}

/// Build positions in parallel while preserving input order.
pub fn build_batch(
    positions: &[engine::Position],
    cfg: &ActBuilderConfig,
) -> Result<Vec<ActGraph>, String> {
    cfg.validate()?;
    positions
        .par_iter()
        .map(|position| build_validated(position, cfg))
        .collect()
}

/// Build positions in parallel and collate them without materializing Python graphs.
pub fn build_packed_batch(
    positions: &[engine::Position],
    cfg: &ActBuilderConfig,
) -> Result<PackedActBatch, String> {
    cfg.validate()?;
    let planned = positions
        .par_iter()
        .map(|position| build_plan(position, cfg))
        .collect::<Result<Vec<_>, _>>()?;
    collate_plans(planned)
}

fn build_prefixes_with<T, F>(
    games: &[Vec<(i16, i16)>],
    ts: &[usize],
    builder: F,
) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(&engine::Position) -> Result<T, String> + Sync,
{
    if games.len() != ts.len() {
        return Err(format!(
            "{} games against {} prefix lengths",
            games.len(),
            ts.len()
        ));
    }
    for (index, (moves, &t)) in games.iter().zip(ts).enumerate() {
        if t > moves.len() {
            return Err(format!(
                "prefix {index} asks for {t} moves of a {}-move game",
                moves.len()
            ));
        }
    }
    games
        .par_iter()
        .zip(ts)
        .enumerate()
        .map(|(index, (moves, &t))| {
            let actions: Vec<_> = moves[..t]
                .iter()
                .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
                .collect();
            let position = engine::Position::replay(&actions)
                .map_err(|error| format!("prefix {index}: {error}"))?;
            builder(&position).map_err(|error| format!("prefix {index}: {error}"))
        })
        .collect()
}

/// Replay each game's first `ts[i]` moves and build every graph in parallel.
pub fn build_batch_prefixes(
    games: &[Vec<(i16, i16)>],
    ts: &[usize],
    cfg: &ActBuilderConfig,
) -> Result<Vec<ActGraph>, String> {
    cfg.validate()?;
    build_prefixes_with(games, ts, |position| build_validated(position, cfg))
}

/// Replay prefixes in parallel and collate them into the model's batch frame.
pub fn build_packed_batch_prefixes(
    games: &[Vec<(i16, i16)>],
    ts: &[usize],
    cfg: &ActBuilderConfig,
) -> Result<PackedActBatch, String> {
    cfg.validate()?;
    let planned = build_prefixes_with(games, ts, |position| build_plan(position, cfg))?;
    collate_plans(planned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_config() -> ActBuilderConfig {
        ActBuilderConfig {
            window_scope: WindowScope::Nonempty,
            cell_scope: CellScope::WindowAndLegal,
            d6_relation_mode: D6RelationMode::Orbit48,
            d_max: 12,
            occupied_radius: 12,
            use_cell_adjacency: true,
            use_occupied_radius_edges: true,
            use_global_numeric_features: true,
            use_window_numeric_features: true,
            use_action_tactical_features: true,
        }
    }

    fn replay(moves: &[(i16, i16)]) -> engine::Position {
        let actions: Vec<_> = moves
            .iter()
            .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
            .collect();
        engine::Position::replay(&actions).expect("legal test position")
    }

    #[test]
    fn ternary_reversal_and_pattern_classes_match_the_frozen_quotient() {
        let tables = pattern_tables();
        let mut representatives = BTreeSet::new();
        for code in 0..TERNARY_CODES {
            let expected_reverse: usize = (0..WINDOW_LEN)
                .map(|slot| digit(code, WINDOW_LEN - 1 - slot) as usize * POWERS[slot])
                .sum();
            assert_eq!(tables.reverse_code[code] as usize, expected_reverse);
            assert_eq!(tables.reverse_code[expected_reverse] as usize, code);
            let representative = code.min(expected_reverse);
            representatives.insert(representative);
            assert_eq!(
                tables.pattern_class[code],
                tables.pattern_class[representative]
            );
        }
        assert_eq!(representatives.len(), 378);
        assert_eq!(
            tables
                .pattern_class
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            (0..378).collect()
        );
        assert_eq!(
            tables.pattern_class[1..]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            377
        );
        for (code, reverse, class) in [
            (0, 0, 0),
            (1, 243, 1),
            (2, 486, 2),
            (3, 81, 3),
            (121, 363, 112),
            (242, 726, 206),
            (364, 364, 278),
            (728, 728, 377),
        ] {
            assert_eq!(tables.reverse_code[code], reverse);
            assert_eq!(tables.pattern_class[code], class);
        }
    }

    #[test]
    fn cell_window_classes_are_the_exact_joint_pair_orbits() {
        let tables = pattern_tables();
        let mut members: FxHashMap<i64, Vec<usize>> = FxHashMap::default();
        for pair in 0..CODE_SLOT_PAIRS {
            let code = pair / WINDOW_LEN;
            let slot = pair % WINDOW_LEN;
            let partner = tables.reverse_code[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
            let class = tables.cell_window_class[pair];
            assert_eq!(class, tables.cell_window_class[partner]);
            members.entry(class).or_default().push(pair);
        }
        assert_eq!(members.len(), 2187);
        assert!(members.values().all(|items| items.len() == 2));
        assert_eq!(
            members.keys().copied().collect::<BTreeSet<_>>(),
            (0..2187).collect()
        );
        assert_eq!(
            tables.cell_window_class[WINDOW_LEN..]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            2184
        );
        assert_eq!(&tables.cell_window_class[..WINDOW_LEN], &[0, 1, 2, 2, 1, 0]);
        for (code, slot, class) in [
            (1, 0, 3),
            (1, 5, 8),
            (364, 0, 1629),
            (364, 5, 1629),
            (728, 0, 2184),
            (728, 5, 2184),
        ] {
            assert_eq!(tables.cell_window_class[code * WINDOW_LEN + slot], class);
        }
    }

    #[test]
    fn post1_classes_cover_exactly_the_own_candidate_pairs() {
        let tables = pattern_tables();
        let mut members: FxHashMap<i64, Vec<usize>> = FxHashMap::default();
        let mut valid = 0;
        for pair in 0..CODE_SLOT_PAIRS {
            let code = pair / WINDOW_LEN;
            let slot = pair % WINDOW_LEN;
            let class = tables.post1_class[pair];
            assert_eq!(class >= 0, digit(code, slot) == OWN);
            if class < 0 {
                continue;
            }
            valid += 1;
            let partner = tables.reverse_code[code] as usize * WINDOW_LEN + WINDOW_LEN - 1 - slot;
            assert_eq!(class, tables.post1_class[partner]);
            members.entry(class).or_default().push(pair);
        }
        assert_eq!(valid, 1458);
        assert_eq!(members.len(), 729);
        assert!(members.values().all(|items| items.len() == 2));
        assert_eq!(
            members.keys().copied().collect::<BTreeSet<_>>(),
            (0..729).collect()
        );
        for (code, slot, class) in [
            (1, 0, 0),
            (243, 5, 0),
            (3, 1, 1),
            (81, 4, 1),
            (9, 2, 6),
            (27, 3, 6),
            (364, 0, 537),
            (364, 5, 537),
        ] {
            assert_eq!(tables.post1_class[code * WINDOW_LEN + slot], class);
        }
        assert!(
            tables.post1_class[..WINDOW_LEN]
                .iter()
                .all(|&class| class == -1)
        );
        assert_eq!(tables.post1_class[365 * WINDOW_LEN], -1);
    }

    #[test]
    fn derived_pattern_tables_match_independent_digit_walks() {
        let tables = pattern_tables();
        for code in 0..TERNARY_CODES {
            let digits: Vec<_> = (0..WINDOW_LEN).map(|slot| digit(code, slot)).collect();
            let own = digits.iter().filter(|&&value| value == OWN).count() as u8;
            let opponent = digits.iter().filter(|&&value| value == OPP).count() as u8;
            let longest = |wanted| {
                digits
                    .split(|&value| value != wanted)
                    .map(<[u8]>::len)
                    .max()
                    .unwrap_or(0) as u8
            };
            assert_eq!(tables.own_count[code], own);
            assert_eq!(tables.opp_count[code], opponent);
            assert_eq!(tables.empty_count[code], 6 - own - opponent);
            assert_eq!(tables.own_max_run[code], longest(OWN));
            assert_eq!(tables.opp_max_run[code], longest(OPP));
            assert_eq!(
                tables.status[code],
                u8::from(own > 0) + 2 * u8::from(opponent > 0)
            );
        }
    }

    #[test]
    fn radius_twelve_has_the_frozen_48_orbits_in_spec_order() {
        let mut representatives = BTreeSet::new();
        let mut ids = BTreeSet::new();
        let mut displacements = 0;
        for dq in -12..=12 {
            for dr in -12..=12 {
                let distance = hex_distance_components(dq, dr);
                if !(1..=12).contains(&distance) {
                    continue;
                }
                displacements += 1;
                let canonical = canonical_displacement((dq, dr));
                representatives.insert((distance, canonical.0, canonical.1));
                let id = orbit_id(dq, dr, 12).expect("inside radius twelve");
                ids.insert(id);
                for reflected in [false, true] {
                    let mut image = if reflected {
                        reflect((dq, dr))
                    } else {
                        (dq, dr)
                    };
                    for _ in 0..6 {
                        assert_eq!(orbit_id(image.0, image.1, 12).unwrap(), id);
                        image = rotate(image);
                    }
                }
            }
        }
        assert_eq!(displacements, 468);
        assert_eq!(representatives.len(), 48);
        assert_eq!(ids, (0..48).collect());
        let expected: Vec<_> = (1..=12)
            .flat_map(|distance| {
                (0..=distance / 2).map(move |r| (distance, -(distance as i16), r as i16))
            })
            .collect();
        assert_eq!(representatives.into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(orbit_id(-1, 0, 12), Ok(0));
        assert_eq!(orbit_id(-6, 3, 12), Ok(14));
        assert_eq!(orbit_id(-12, 6, 12), Ok(47));
        assert_eq!(
            orbit_id(0, 0, 12).unwrap_err(),
            "displacement (0, 0) has hex distance 0, outside the 1..12 orbit table"
        );
    }

    #[test]
    fn radius_six_uses_the_exact_prefix_and_axis_routes_are_structural() {
        let ids: BTreeSet<_> = (-6..=6)
            .flat_map(|dq| (-6..=6).map(move |dr| (dq, dr)))
            .filter(|&(dq, dr)| (1..=6).contains(&hex_distance_components(dq, dr)))
            .map(|(dq, dr)| orbit_id(dq, dr, 6).unwrap())
            .collect();
        assert_eq!(ids, (0..15).collect());
        assert_eq!(on_axis(0, 0), -1);
        assert_eq!(on_axis(5, 0), 0);
        assert_eq!(on_axis(0, -5), 1);
        assert_eq!(on_axis(-5, 5), 2);
        assert_eq!(on_axis(2, 1), -1);
    }

    #[test]
    fn opening_graph_pins_action_rows_and_both_window_scopes() {
        let opening = engine::Position::new();
        let graph = build(&opening, &full_config()).expect("opening is live");
        assert_eq!(graph.cell_qr, [0, 0]);
        assert_eq!(graph.cell_occupancy, [0]);
        assert_eq!(graph.cell_is_legal, [1]);
        assert_eq!(graph.cell_nearest_bucket, [NEAREST_UNREACHED]);
        assert_eq!(graph.legal_to_cell_index, [0]);
        assert_eq!(graph.n_windows(), 0);
        assert_eq!(graph.action_window_index, [-1; POST_ACTION_ROWS]);
        assert_eq!(graph.action_pre_status, [0; POST_ACTION_ROWS]);
        for axis in 0..NUM_AXES {
            assert_eq!(
                &graph.action_post1_class[axis * WINDOW_LEN..(axis + 1) * WINDOW_LEN],
                &[0, 1, 6, 6, 1, 0]
            );
        }
        assert_eq!(graph.phase_id, 0);
        assert_eq!(graph.moves_remaining, 1);

        let mut relevant = full_config();
        relevant.window_scope = WindowScope::ActionRelevant;
        let graph = build(&opening, &relevant).expect("opening is live");
        assert_eq!(graph.n_windows(), 18);
        assert_eq!(graph.n_cells(), 31);
        // Besides the ten links along each of the three lines, the six cells
        // one step from the origin form a hexagon around it.
        assert_eq!(graph.adjacency_src.len(), 72);
        assert!(graph.radius_src.is_empty());
        assert_eq!(
            graph.action_window_index,
            [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 17, 16, 15, 14, 13, 12,]
        );
        assert!(graph.window_pattern_class.iter().all(|&class| class == 0));
        assert!(graph.window_status.iter().all(|&status| status == 0));
        assert!(
            graph
                .window_numeric
                .chunks_exact(WINDOW_NUMERIC_FEATURES)
                .all(|row| row == [0.0, 0.0, 1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn auxiliary_labels_cover_the_opening_and_crafted_threat_positions() {
        let opening = build_aux_labels(&engine::Position::new(), &full_config()).unwrap();
        assert_eq!(opening.win_now, vec![0]);
        assert_eq!(opening.own_max_occupancy, vec![1]);
        assert_eq!(opening.opponent_threats_hit, vec![0]);
        assert_eq!(opening.own_five_windows_after, vec![0]);
        assert_eq!(opening.winning_partner_exists, vec![0]);
        assert_eq!(opening.winning_partner_count, vec![0]);

        let threat_game = [
            (0, 0),
            (0, 7),
            (1, 7),
            (1, 0),
            (2, 0),
            (2, 7),
            (3, 7),
            (3, 0),
            (3, 3),
        ];
        let win_game = [threat_game.as_slice(), &[(4, 7)]].concat();
        let labels = [
            build_aux_labels(&replay(&threat_game), &full_config()).unwrap(),
            build_aux_labels(&replay(&win_game), &full_config()).unwrap(),
        ];
        for label in &labels {
            let legal = label.win_now.len();
            assert!(legal > 0);
            assert_eq!(label.own_max_occupancy.len(), legal);
            assert_eq!(label.opponent_threats_hit.len(), legal);
            assert_eq!(label.own_five_windows_after.len(), legal);
            assert_eq!(label.winning_partner_exists.len(), legal);
            assert_eq!(label.winning_partner_count.len(), legal);
            assert!(label.win_now.iter().all(|&value| (0..=1).contains(&value)));
            assert!(
                label
                    .own_max_occupancy
                    .iter()
                    .all(|&value| (1..=6).contains(&value))
            );
            for values in [
                &label.opponent_threats_hit,
                &label.own_five_windows_after,
                &label.winning_partner_count,
            ] {
                assert!(
                    values
                        .iter()
                        .all(|&value| (0..=AUX_COUNT_CAP as i64).contains(&value))
                );
            }
        }
        assert!(labels.iter().any(|label| label.win_now.contains(&1)));
        assert!(
            labels
                .iter()
                .any(|label| label.opponent_threats_hit.iter().any(|&value| value > 0))
        );
        assert!(
            labels
                .iter()
                .any(|label| label.own_five_windows_after.iter().any(|&value| value > 0))
        );
        assert!(
            labels
                .iter()
                .any(|label| label.winning_partner_exists.contains(&1))
        );
    }

    #[test]
    fn every_emitted_family_obeys_the_section_seven_order() {
        let position = replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]);
        let graph = build(&position, &full_config()).expect("fixture is live");
        assert!(
            graph
                .cell_qr
                .chunks_exact(2)
                .map(|row| (row[0], row[1]))
                .collect::<Vec<_>>()
                .windows(2)
                .all(|rows| rows[0] < rows[1])
        );
        assert!(
            graph
                .window_id
                .chunks_exact(3)
                .map(|row| (row[0], row[1], row[2]))
                .collect::<Vec<_>>()
                .windows(2)
                .all(|rows| rows[0] < rows[1])
        );
        assert!(
            (0..graph.adjacency_src.len())
                .collect::<Vec<_>>()
                .windows(2)
                .all(|rows| {
                    let key = |row: usize| {
                        (
                            graph.adjacency_dst[row],
                            graph.adjacency_src[row],
                            graph.adjacency_axis[row],
                        )
                    };
                    key(rows[0]) <= key(rows[1])
                })
        );
        assert!(
            (0..graph.radius_src.len())
                .collect::<Vec<_>>()
                .windows(2)
                .all(|rows| {
                    let key = |row: usize| {
                        (
                            graph.radius_dst[row],
                            graph.radius_src[row],
                            graph.radius_orbit[row],
                        )
                    };
                    key(rows[0]) <= key(rows[1])
                })
        );

        let legal: Vec<_> = position
            .legal_actions()
            .map(|action| action.coord())
            .collect();
        for (&cell, expected) in graph.legal_to_cell_index.iter().zip(legal) {
            let row = cell as usize * 2;
            assert_eq!(
                (graph.cell_qr[row], graph.cell_qr[row + 1]),
                (i64::from(expected.q), i64::from(expected.r))
            );
        }
    }

    fn naive_radius_rows(
        position: &engine::Position,
        graph: &ActGraph,
        config: &ActBuilderConfig,
    ) -> Vec<(i64, i64, i64, i64)> {
        let cell_index: FxHashMap<_, _> = graph
            .cell_qr
            .chunks_exact(2)
            .enumerate()
            .map(|(index, row)| {
                (
                    engine::HexCoord::new(row[0] as i16, row[1] as i16),
                    index as i64,
                )
            })
            .collect();
        let radius = config.occupied_radius as i16;
        let mut rows = Vec::new();
        for (stone, _) in position.stones() {
            let source = cell_index[&stone];
            for dq in -radius..=radius {
                for dr in -radius..=radius {
                    let distance = hex_distance_components(dq, dr);
                    if !(1..=config.occupied_radius).contains(&distance) {
                        continue;
                    }
                    let coord = offset_coord(stone, dq, dr, 1);
                    let Some(&destination) = cell_index.get(&coord) else {
                        continue;
                    };
                    let (relation, route) = radius_relation(dq, dr, distance, config).unwrap();
                    rows.push((destination, source, relation, route));
                }
            }
        }
        rows.sort_unstable_by_key(|&(destination, source, relation, _)| {
            (destination, source, relation)
        });
        rows
    }

    #[test]
    fn radius_edges_match_the_naive_order_in_both_enumeration_regimes() {
        let position = replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]);
        let direct = full_config();
        let mut binned = direct;
        binned.d6_relation_mode = D6RelationMode::CoarseDistanceAxis;
        binned.occupied_radius = 1;

        for config in [direct, binned] {
            let graph = build(&position, &config).unwrap();
            let diameter = 2 * config.occupied_radius + 1;
            let uses_direct_enumeration = graph.n_cells() <= diameter * diameter;
            assert_eq!(uses_direct_enumeration, config.occupied_radius == 12);
            let actual: Vec<_> = (0..graph.radius_src.len())
                .map(|row| {
                    (
                        graph.radius_dst[row],
                        graph.radius_src[row],
                        graph.radius_orbit[row],
                        graph.radius_axis_or_neg1[row],
                    )
                })
                .collect();
            assert_eq!(actual, naive_radius_rows(&position, &graph, &config));
        }
    }

    #[test]
    fn toggles_remove_only_their_families_or_feature_widths() {
        let position = replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]);
        let mut config = full_config();
        config.use_cell_adjacency = false;
        config.use_occupied_radius_edges = false;
        config.use_global_numeric_features = false;
        config.use_window_numeric_features = false;
        config.use_action_tactical_features = false;
        let graph = build(&position, &config).expect("fixture is live");
        assert!(graph.adjacency_src.is_empty());
        assert!(graph.radius_src.is_empty());
        assert!(graph.global_numeric.is_empty());
        assert_eq!(graph.global_numeric_width, 0);
        assert!(graph.window_numeric.is_empty());
        assert_eq!(graph.window_numeric_width, 0);
        assert!(graph.action_tactical_numeric.is_empty());
        assert_eq!(graph.action_tactical_numeric_width, 0);
        assert!(graph.n_cells() > 0);
        assert!(graph.n_windows() > 0);
        assert!(graph.n_legal() > 0);
    }

    #[test]
    fn prefix_and_direct_parallel_entry_points_share_the_same_core() {
        let games = vec![vec![], vec![(0, 0)], vec![(0, 0), (1, 0), (2, 0), (0, 1)]];
        let ts = vec![0, 1, 4];
        let positions: Vec<_> = games
            .iter()
            .zip(&ts)
            .map(|(moves, &t)| replay(&moves[..t]))
            .collect();
        let direct = build_batch(&positions, &full_config()).unwrap();
        let prefixes = build_batch_prefixes(&games, &ts, &full_config()).unwrap();
        assert_eq!(direct, prefixes);
        assert_eq!(
            build_packed_batch(&positions, &full_config()).unwrap(),
            build_packed_batch_prefixes(&games, &ts, &full_config()).unwrap()
        );
    }

    #[test]
    fn direct_packed_plans_match_public_graph_collation_in_every_regime() {
        let positions = vec![
            replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]),
            replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]),
        ];
        let direct = full_config();
        let mut binned_orbit = direct;
        binned_orbit.occupied_radius = 1;
        let mut binned = direct;
        binned.d6_relation_mode = D6RelationMode::CoarseDistanceAxis;
        binned.occupied_radius = 1;
        let mut disabled = direct;
        disabled.use_occupied_radius_edges = false;
        let mut live = direct;
        live.window_scope = WindowScope::Live;
        let mut action_relevant = direct;
        action_relevant.window_scope = WindowScope::ActionRelevant;

        for config in [
            direct,
            binned_orbit,
            binned,
            disabled,
            live,
            action_relevant,
        ] {
            let expected = collate(build_batch(&positions, &config).unwrap()).unwrap();
            let actual = build_packed_batch(&positions, &config).unwrap();
            assert_eq!(actual, expected);
        }
    }

    fn shifted(local: &[i64], offset: i64, sentinel: bool) -> Vec<i64> {
        local
            .iter()
            .map(|&value| {
                if sentinel && value == -1 {
                    -1
                } else {
                    value + offset
                }
            })
            .collect()
    }

    fn assert_position_local(
        values: &[i64],
        row_offsets: &[i64],
        target_offsets: &[i64],
        width: usize,
        sentinel: bool,
    ) {
        for position in 0..row_offsets.len() - 1 {
            let row_start = row_offsets[position] as usize * width;
            let row_end = row_offsets[position + 1] as usize * width;
            let target_start = target_offsets[position];
            let target_end = target_offsets[position + 1];
            for &value in &values[row_start..row_end] {
                assert!(
                    (sentinel && value == -1) || (target_start <= value && value < target_end),
                    "position {position} index {value} is outside [{target_start}, {target_end})"
                );
            }
        }
    }

    #[test]
    fn rust_collation_shifts_every_index_by_its_target_family_offset() {
        let first = build(&replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]), &full_config()).unwrap();
        let second = build(
            &replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]),
            &full_config(),
        )
        .unwrap();

        let first_cells = first.n_cells() as i64;
        let first_windows = first.n_windows() as i64;
        let expected_cells = [first.cell_occupancy.clone(), second.cell_occupancy.clone()].concat();
        let expected_windows = [first.window_id.clone(), second.window_id.clone()].concat();
        let expected_legal = [
            first.legal_to_cell_index.clone(),
            shifted(&second.legal_to_cell_index, first_cells, true),
        ]
        .concat();
        let expected_window_cells = [
            first.window_cell_index.clone(),
            shifted(&second.window_cell_index, first_cells, true),
        ]
        .concat();
        let expected_action_windows = [
            first.action_window_index.clone(),
            shifted(&second.action_window_index, first_windows, true),
        ]
        .concat();
        let first_radius_bound = first.radius_orbit.iter().copied().max().unwrap_or(-1);
        let second_radius_bound = second.radius_orbit.iter().copied().max().unwrap_or(-1);
        let expected_radius_bound = first_radius_bound.max(second_radius_bound) + 1;
        let expected_offsets = (
            vec![
                0,
                first.n_cells() as i64,
                (first.n_cells() + second.n_cells()) as i64,
            ],
            vec![
                0,
                first.n_windows() as i64,
                (first.n_windows() + second.n_windows()) as i64,
            ],
            vec![
                0,
                first.n_legal() as i64,
                (first.n_legal() + second.n_legal()) as i64,
            ],
        );

        let packed = collate(vec![first, second]).unwrap();
        assert_eq!(packed.position_count, 2);
        assert_eq!(packed.cell_offsets, expected_offsets.0);
        assert_eq!(packed.window_offsets, expected_offsets.1);
        assert_eq!(packed.legal_offsets, expected_offsets.2);
        assert_eq!(packed.cell_occupancy, expected_cells);
        assert_eq!(packed.window_id, expected_windows);
        assert_eq!(packed.legal_to_cell_index, expected_legal);
        assert_eq!(packed.window_cell_index, expected_window_cells);
        assert_eq!(packed.action_window_index, expected_action_windows);
        assert_eq!(packed.radius_orbit_bound, expected_radius_bound);
    }

    #[test]
    fn rust_collation_guarantees_every_index_stays_inside_its_position() {
        let graphs = vec![
            build(&replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]), &full_config()).unwrap(),
            build(
                &replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]),
                &full_config(),
            )
            .unwrap(),
        ];
        let packed = collate(graphs).unwrap();

        assert_position_local(
            &packed.legal_to_cell_index,
            &packed.legal_offsets,
            &packed.cell_offsets,
            1,
            true,
        );
        assert_position_local(
            &packed.window_cell_index,
            &packed.window_offsets,
            &packed.cell_offsets,
            WINDOW_LEN,
            true,
        );
        for values in [&packed.adjacency_src, &packed.adjacency_dst] {
            assert_position_local(
                values,
                &packed.adjacency_offsets,
                &packed.cell_offsets,
                1,
                false,
            );
        }
        for values in [&packed.radius_src, &packed.radius_dst] {
            assert_position_local(
                values,
                &packed.radius_offsets,
                &packed.cell_offsets,
                1,
                false,
            );
        }
        assert_position_local(
            &packed.action_window_index,
            &packed.legal_offsets,
            &packed.window_offsets,
            POST_ACTION_ROWS,
            true,
        );
    }

    #[test]
    fn rust_collation_refuses_a_local_index_outside_its_target_family() {
        let mut first = build(&replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]), &full_config()).unwrap();
        let second = build(
            &replay(&[(0, 0), (3, 0), (-2, 2), (0, 3), (1, -2), (-1, 3)]),
            &full_config(),
        )
        .unwrap();
        let first_cell_count = first.n_cells() as i64;
        first.adjacency_dst[0] = first_cell_count;

        let error = collate(vec![first, second]).unwrap_err();
        assert_eq!(
            error,
            format!(
                "adjacency_dst contains local index {first_cell_count} outside its target family's 0..{first_cell_count} range"
            )
        );
    }

    #[test]
    fn rust_collation_refuses_empty_or_mixed_width_batches() {
        assert_eq!(
            collate(Vec::new()).unwrap_err(),
            "empty batch: collate needs at least one position"
        );

        let position = replay(&[(0, 0)]);
        let first = build(&position, &full_config()).unwrap();
        let mut without_numeric = full_config();
        without_numeric.use_window_numeric_features = false;
        let second = build(&position, &without_numeric).unwrap();
        assert!(
            collate(vec![first, second])
                .unwrap_err()
                .contains("window_numeric has inconsistent feature widths")
        );
    }

    #[test]
    fn rust_collation_refuses_malformed_column_lengths() {
        let mut graph = build(&replay(&[(0, 0), (1, 0)]), &full_config()).unwrap();
        let expected = graph.n_cells();
        graph.cell_is_legal.pop();

        assert_eq!(
            collate(vec![graph]).unwrap_err(),
            format!(
                "position 0 cell_is_legal has {} values, expected {expected}",
                expected - 1
            )
        );
    }

    #[test]
    fn planned_radius_refuses_graph_cell_count_drift() {
        let position = replay(&[(0, 0), (1, 0)]);
        let (graph, mut plan, action_plan) = build_plan(&position, &full_config()).unwrap();
        let expected = graph.n_cells();
        plan.as_mut().expect("radius edges are enabled").cell_count += 1;

        assert_eq!(
            collate_plans(vec![(graph, plan, action_plan)]).unwrap_err(),
            format!(
                "position 0 radius plan has {} cells against graph's {expected}",
                expected + 1
            )
        );
    }

    #[test]
    fn planned_action_refuses_graph_count_drift_and_malformed_columns() {
        let position = replay(&[(0, 0), (1, 0)]);
        let (graph, radius_plan, mut action_plan) = build_plan(&position, &full_config()).unwrap();
        let expected = graph.n_legal();
        let windows = graph.n_windows();
        action_plan.legal_count += 1;
        assert_eq!(
            collate_plans(vec![(graph, radius_plan, action_plan)]).unwrap_err(),
            format!(
                "position 0 action plan has {windows} windows and {} legal rows against graph's {windows} and {expected}",
                expected + 1
            )
        );

        let (graph, radius_plan, mut action_plan) = build_plan(&position, &full_config()).unwrap();
        action_plan.pre_status.pop();
        let expected_rows = expected * POST_ACTION_ROWS;
        assert_eq!(
            collate_plans(vec![(graph, radius_plan, action_plan)]).unwrap_err(),
            format!(
                "position 0: action plan for {expected} legal rows has {expected_rows}, {expected_rows}, and {} values; expected {expected_rows}",
                expected_rows - 1
            )
        );
    }

    #[test]
    fn compact_action_plan_refuses_every_out_of_range_value() {
        fn fill_error(plan: &ActionPlan) -> String {
            let rows = plan.rows();
            fill_action_rows(
                plan,
                0,
                (&mut vec![0; rows], &mut vec![0; rows], &mut vec![0; rows]),
            )
            .unwrap_err()
        }

        let position = replay(&[(0, 0), (1, 0)]);
        let (_, _, mut plan) = build_plan(&position, &full_config()).unwrap();
        plan.window_index[0] = -2;
        assert_eq!(
            fill_error(&plan),
            format!(
                "action plan row 0 indexes window -2 against {} windows",
                plan.window_count
            )
        );

        let (_, _, mut plan) = build_plan(&position, &full_config()).unwrap();
        plan.window_index[0] = i32::try_from(plan.window_count).unwrap();
        assert_eq!(
            fill_error(&plan),
            format!(
                "action plan row 0 indexes window {} against {} windows",
                plan.window_count, plan.window_count
            )
        );

        let (_, _, mut plan) = build_plan(&position, &full_config()).unwrap();
        plan.post1_class[0] = EXPECTED_POST1_CLASSES as u16;
        assert_eq!(
            fill_error(&plan),
            "action plan row 0 has post1 class 729 outside 0..729"
        );

        let (_, _, mut plan) = build_plan(&position, &full_config()).unwrap();
        plan.pre_status[0] = MIXED + 1;
        assert_eq!(
            fill_error(&plan),
            "action plan row 0 has pre-status 4 outside 0..=3"
        );
    }

    #[test]
    fn invalid_configs_and_prefix_shapes_fail_loudly() {
        let mut config = full_config();
        config.d_max = 13;
        assert!(config.validate().unwrap_err().contains("at most 12"));
        config.d6_relation_mode = D6RelationMode::CoarseDistanceAxis;
        config.occupied_radius = 14;
        assert!(config.validate().unwrap_err().contains("exceeds d_max"));
        config.occupied_radius = 0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("disable use_occupied_radius_edges")
        );

        assert_eq!(
            build_batch_prefixes(&[vec![(0, 0)]], &[], &full_config()).unwrap_err(),
            "1 games against 0 prefix lengths"
        );
        assert_eq!(
            build_batch_prefixes(&[vec![(0, 0)]], &[2], &full_config()).unwrap_err(),
            "prefix 0 asks for 2 moves of a 1-move game"
        );
    }
}
