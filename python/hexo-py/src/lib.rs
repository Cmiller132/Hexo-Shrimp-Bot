//! Python bindings for `hexo-engine`: the read surface a model builder needs,
//! and nothing that could bypass the rules.
//!
//! The surface is the input list — stones, legal moves in canonical order,
//! `moves_remaining` — plus `windows_through`, which exists so a builder test
//! can check window enumeration against the engine as an independent oracle.
//! Positions are created empty or by replay, never deserialised: a board-shaped
//! constructor would be a rule-bypass hole, which is the same argument the
//! engine makes for itself.

use hexo_engine as engine;
use hexo_model_mantisnet::{MODEL_REPR_VERSION, act_encoder, encoder};
use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};

// Packed ACT arrays outlive this call as zero-copy NumPy/Torch storage.  The
// fitting worker later releases four dominant radius allocations together;
// an allocator designed for concurrent page reuse avoids making eager system
// purging the extension's policy on that producer critical path.  This
// allocator is local to Rust allocations made by this extension.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A Hexo position. Wraps `hexo_engine::Position` one-to-one.
#[pyclass]
struct Position {
    inner: engine::Position,
}

/// One window through a cell: `(axis, start_q, start_r, mask_p0, mask_p1)`.
type WindowTuple = (u8, i16, i16, u8, u8);

fn action(q: i16, r: i16) -> engine::Action {
    engine::Action::new(engine::HexCoord::new(q, r))
}

#[pymethods]
impl Position {
    /// The empty position: `P0` to move at the origin.
    #[new]
    fn new() -> Self {
        Self {
            inner: engine::Position::new(),
        }
    }

    /// Replay a placement sequence from the empty board.
    #[staticmethod]
    fn replay(moves: Vec<(i16, i16)>) -> PyResult<Self> {
        let actions: Vec<engine::Action> = moves.iter().map(|&(q, r)| action(q, r)).collect();
        engine::Position::replay(&actions)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply one placement. Raises `ValueError` on an illegal move.
    fn advance(&mut self, q: i16, r: i16) -> PyResult<()> {
        self.inner
            .advance(action(q, r))
            .map(|_| ())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// An independent copy.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Every stone as `(q, r, player)`, in canonical `(q, r)` order.
    fn stones(&self) -> Vec<(i16, i16, u8)> {
        self.inner
            .stones()
            .map(|(c, p)| (c.q, c.r, p.index() as u8))
            .collect()
    }

    /// Legal placements as `(q, r)`, in the engine's canonical order
    /// (`ACTION_ORDER_VERSION`). Empty exactly when the position is terminal.
    fn legal_moves(&self) -> Vec<(i16, i16)> {
        self.inner
            .legal_actions()
            .map(|a| {
                let c = a.coord();
                (c.q, c.r)
            })
            .collect()
    }

    /// The legal placement at `index` in that same order — what a caller
    /// holding a sampled rank wants, without materialising the whole list.
    fn nth_legal(&self, index: usize) -> PyResult<(i16, i16)> {
        self.inner
            .nth_legal(index)
            .map(|a| {
                let c = a.coord();
                (c.q, c.r)
            })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "legal index {index} out of range ({} legal moves)",
                    self.inner.legal_count()
                ))
            })
    }

    /// The 18 win windows through `(q, r)`: `(axis, start_q, start_r, mask_p0,
    /// mask_p1)`, where bit `k` of a mask is the cell `k` steps from the start
    /// along the axis. Axes are `0 = Q (1,0)`, `1 = R (0,1)`, `2 = QR (1,-1)`.
    ///
    /// This is the engine's own window walk, exposed so Python tests and
    /// diagnostics can inspect the authoritative geometry directly.
    fn windows_through(&self, q: i16, r: i16) -> PyResult<Vec<WindowTuple>> {
        let c = engine::HexCoord::new(q, r);
        if !c.is_valid() {
            return Err(PyValueError::new_err(format!(
                "coordinate ({q}, {r}) is outside the engine's domain"
            )));
        }
        Ok(self
            .inner
            .windows_through(c)
            .iter()
            .filter(|wr| wr.window.start.is_valid())
            .map(|wr| {
                (
                    wr.window.axis.index() as u8,
                    wr.window.start.q,
                    wr.window.start.r,
                    wr.mask.mask(engine::Player::P0),
                    wr.mask.mask(engine::Player::P1),
                )
            })
            .collect())
    }

    /// Number of legal placements. `0` if and only if terminal.
    #[getter]
    fn legal_count(&self) -> usize {
        self.inner.legal_count()
    }

    /// Whose turn it is: `0` or `1`. Frozen at the winner once terminal.
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player().index() as u8
    }

    /// Placements the mover still has this turn: `2` before the first stone of
    /// a normal turn, `1` before its second stone or the opening stone.
    #[getter]
    fn moves_remaining(&self) -> u8 {
        match self.inner.phase() {
            engine::TurnPhase::FirstStone => 2,
            engine::TurnPhase::Opening | engine::TurnPhase::SecondStone => 1,
        }
    }

    /// Whether the game is over.
    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// The winner (`0` or `1`), or `None` while the game runs.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.outcome().map(|o| o.winner.index() as u8)
    }

    /// Total stones placed.
    #[getter]
    fn stone_count(&self) -> u32 {
        self.inner.stone_count()
    }

    /// Incremental Zobrist hash.
    #[getter]
    fn zobrist(&self) -> u64 {
        self.inner.zobrist()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Position stones={} to_move={} terminal={}>",
            self.inner.stone_count(),
            self.inner.current_player().index(),
            self.inner.is_terminal(),
        )
    }
}

/// A collated `RawBatch` as a dict of numpy arrays, keyed by the field names
/// `mantisnet.builder.Batch` uses.
fn raw_to_dict<'py>(py: Python<'py>, raw: encoder::RawBatch) -> PyResult<Bound<'py, PyDict>> {
    let (p, max_t, max_w) = (raw.n_pos, raw.max_t, raw.max_w);
    let d = PyDict::new(py);
    let n_w = raw.window_feat.len();
    d.set_item("stone_own", PyArray1::from_vec(py, raw.stone_own))?;
    d.set_item("window_feat", PyArray1::from_vec(py, raw.window_feat))?;
    d.set_item(
        "window_id",
        PyArray1::from_vec(py, raw.window_id).reshape([n_w, 3])?,
    )?;
    d.set_item("moves_idx", PyArray1::from_vec(py, raw.moves_idx))?;
    d.set_item("inc_stone", PyArray1::from_vec(py, raw.inc_stone))?;
    d.set_item("inc_window", PyArray1::from_vec(py, raw.inc_window))?;
    d.set_item("inc_class", PyArray1::from_vec(py, raw.inc_class))?;
    d.set_item("stone_slot", PyArray1::from_vec(py, raw.stone_slot))?;
    d.set_item(
        "coords",
        PyArray1::from_vec(py, raw.coords).reshape([p, max_t, 2])?,
    )?;
    d.set_item(
        "attn_valid",
        PyArray1::from_vec(py, raw.attn_valid).reshape([p, max_t])?,
    )?;
    d.set_item("window_slot", PyArray1::from_vec(py, raw.window_slot))?;
    d.set_item(
        "value_valid",
        PyArray1::from_vec(py, raw.value_valid).reshape([p, max_w])?,
    )?;
    d.set_item("legal_offsets", PyArray1::from_vec(py, raw.legal_offsets))?;
    d.set_item("cell_pos", PyArray1::from_vec(py, raw.cell_pos))?;
    d.set_item("dec_cell", PyArray1::from_vec(py, raw.dec_cell))?;
    d.set_item("dec_window", PyArray1::from_vec(py, raw.dec_window))?;
    d.set_item("dec_class", PyArray1::from_vec(py, raw.dec_class))?;
    d.set_item("bg_cell", PyArray1::from_vec(py, raw.bg_cell))?;
    d.set_item("bg_bucket", PyArray1::from_vec(py, raw.bg_bucket))?;
    Ok(d)
}

const ACT_CONFIG_KEYS: [&str; 10] = [
    "window_scope",
    "cell_scope",
    "d6_relation_mode",
    "d_max",
    "occupied_radius",
    "use_cell_adjacency",
    "use_occupied_radius_edges",
    "use_global_numeric_features",
    "use_window_numeric_features",
    "use_action_tactical_features",
];

fn required_config_item<'py>(
    config: &Bound<'py, PyDict>,
    name: &'static str,
) -> PyResult<Bound<'py, PyAny>> {
    config
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("ACT builder config is missing {name:?}")))
}

fn config_usize(config: &Bound<'_, PyDict>, name: &'static str) -> PyResult<usize> {
    let value = required_config_item(config, name)?.extract::<i64>()?;
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("ACT builder config {name}={value} is negative"))
    })
}

fn parse_act_config(config: &Bound<'_, PyDict>) -> PyResult<act_encoder::ActBuilderConfig> {
    for key in config.keys().iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("ACT builder config keys must all be strings"))?;
        if !ACT_CONFIG_KEYS.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown ACT builder config field {key:?}"
            )));
        }
    }
    for &key in &ACT_CONFIG_KEYS {
        if !config.contains(key)? {
            return Err(PyValueError::new_err(format!(
                "ACT builder config is missing {key:?}"
            )));
        }
    }

    let window_scope = match required_config_item(config, "window_scope")?.extract::<String>()? {
        value if value == "live" => act_encoder::WindowScope::Live,
        value if value == "nonempty" => act_encoder::WindowScope::Nonempty,
        value if value == "action_relevant" => act_encoder::WindowScope::ActionRelevant,
        value => {
            return Err(PyValueError::new_err(format!(
                "unknown ACT window_scope {value:?}"
            )));
        }
    };
    let cell_scope = match required_config_item(config, "cell_scope")?.extract::<String>()? {
        value if value == "occupied_only" => act_encoder::CellScope::OccupiedOnly,
        value if value == "occupied_and_legal" => act_encoder::CellScope::OccupiedAndLegal,
        value if value == "window_and_legal" => act_encoder::CellScope::WindowAndLegal,
        value => {
            return Err(PyValueError::new_err(format!(
                "unknown ACT cell_scope {value:?}"
            )));
        }
    };
    let d6_relation_mode = required_config_item(config, "d6_relation_mode")?.extract::<String>()?;
    let d6_relation_mode = match d6_relation_mode.as_str() {
        "orbit48" => act_encoder::D6RelationMode::Orbit48,
        "coarse_distance_axis" => act_encoder::D6RelationMode::CoarseDistanceAxis,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown ACT d6_relation_mode {d6_relation_mode:?}"
            )));
        }
    };
    let d_max = config_usize(config, "d_max")?;
    let occupied_radius = config_usize(config, "occupied_radius")?;
    let use_cell_adjacency =
        required_config_item(config, "use_cell_adjacency")?.extract::<bool>()?;
    let use_occupied_radius_edges =
        required_config_item(config, "use_occupied_radius_edges")?.extract::<bool>()?;
    let use_global_numeric_features =
        required_config_item(config, "use_global_numeric_features")?.extract::<bool>()?;
    let use_window_numeric_features =
        required_config_item(config, "use_window_numeric_features")?.extract::<bool>()?;
    let use_action_tactical_features =
        required_config_item(config, "use_action_tactical_features")?.extract::<bool>()?;

    let parsed = act_encoder::ActBuilderConfig {
        window_scope,
        cell_scope,
        d6_relation_mode,
        d_max,
        occupied_radius,
        use_cell_adjacency,
        use_occupied_radius_edges,
        use_global_numeric_features,
        use_window_numeric_features,
        use_action_tactical_features,
    };
    parsed.validate().map_err(PyValueError::new_err)?;
    Ok(parsed)
}

/// One position-local ACT graph as the numpy arrays `ACTGraph` accepts.
fn act_graph_to_dict<'py>(
    py: Python<'py>,
    graph: act_encoder::ActGraph,
) -> PyResult<Bound<'py, PyDict>> {
    let n_cells = graph.cell_occupancy.len();
    let n_windows = graph.window_pattern_class.len();
    let n_legal = graph.legal_to_cell_index.len();
    let window_numeric_width = graph.window_numeric_width;
    let action_tactical_numeric_width = graph.action_tactical_numeric_width;
    let d = PyDict::new(py);
    d.set_item(
        "cell_qr",
        PyArray1::from_vec(py, graph.cell_qr).reshape([n_cells, 2])?,
    )?;
    d.set_item(
        "cell_occupancy",
        PyArray1::from_vec(py, graph.cell_occupancy),
    )?;
    d.set_item("cell_is_legal", PyArray1::from_vec(py, graph.cell_is_legal))?;
    d.set_item(
        "cell_is_occupied",
        PyArray1::from_vec(py, graph.cell_is_occupied),
    )?;
    d.set_item(
        "cell_nearest_bucket",
        PyArray1::from_vec(py, graph.cell_nearest_bucket),
    )?;
    d.set_item(
        "legal_to_cell_index",
        PyArray1::from_vec(py, graph.legal_to_cell_index),
    )?;
    d.set_item(
        "window_id",
        PyArray1::from_vec(py, graph.window_id).reshape([n_windows, 3])?,
    )?;
    d.set_item(
        "window_pattern_class",
        PyArray1::from_vec(py, graph.window_pattern_class),
    )?;
    d.set_item("window_status", PyArray1::from_vec(py, graph.window_status))?;
    d.set_item("window_axis", PyArray1::from_vec(py, graph.window_axis))?;
    d.set_item(
        "window_numeric",
        PyArray1::from_vec(py, graph.window_numeric).reshape([n_windows, window_numeric_width])?,
    )?;
    d.set_item(
        "window_cell_index",
        PyArray1::from_vec(py, graph.window_cell_index).reshape([n_windows, 6])?,
    )?;
    d.set_item(
        "window_incidence_class",
        PyArray1::from_vec(py, graph.window_incidence_class).reshape([n_windows, 6])?,
    )?;
    d.set_item(
        "window_incidence_mask",
        PyArray1::from_vec(py, graph.window_incidence_mask).reshape([n_windows, 6])?,
    )?;
    d.set_item("adjacency_src", PyArray1::from_vec(py, graph.adjacency_src))?;
    d.set_item("adjacency_dst", PyArray1::from_vec(py, graph.adjacency_dst))?;
    d.set_item(
        "adjacency_axis",
        PyArray1::from_vec(py, graph.adjacency_axis),
    )?;
    d.set_item("radius_src", PyArray1::from_vec(py, graph.radius_src))?;
    d.set_item("radius_dst", PyArray1::from_vec(py, graph.radius_dst))?;
    d.set_item("radius_orbit", PyArray1::from_vec(py, graph.radius_orbit))?;
    d.set_item(
        "radius_axis_or_neg1",
        PyArray1::from_vec(py, graph.radius_axis_or_neg1),
    )?;
    d.set_item(
        "action_window_index",
        PyArray1::from_vec(py, graph.action_window_index).reshape([n_legal, 3, 6])?,
    )?;
    d.set_item(
        "action_post1_class",
        PyArray1::from_vec(py, graph.action_post1_class).reshape([n_legal, 3, 6])?,
    )?;
    d.set_item(
        "action_pre_status",
        PyArray1::from_vec(py, graph.action_pre_status).reshape([n_legal, 3, 6])?,
    )?;
    d.set_item(
        "action_tactical_numeric",
        PyArray1::from_vec(py, graph.action_tactical_numeric)
            .reshape([n_legal, action_tactical_numeric_width])?,
    )?;
    d.set_item(
        "global_numeric",
        PyArray1::from_vec(py, graph.global_numeric),
    )?;
    d.set_item("moves_remaining", graph.moves_remaining)?;
    d.set_item("phase_id", graph.phase_id)?;
    Ok(d)
}

/// Section 24.1's six deterministic action-label arrays.
fn act_aux_labels_to_dict<'py>(
    py: Python<'py>,
    labels: act_encoder::ActAuxLabels,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("win_now", PyArray1::from_vec(py, labels.win_now))?;
    d.set_item(
        "own_max_occupancy",
        PyArray1::from_vec(py, labels.own_max_occupancy),
    )?;
    d.set_item(
        "opponent_threats_hit",
        PyArray1::from_vec(py, labels.opponent_threats_hit),
    )?;
    d.set_item(
        "own_five_windows_after",
        PyArray1::from_vec(py, labels.own_five_windows_after),
    )?;
    d.set_item(
        "winning_partner_exists",
        PyArray1::from_vec(py, labels.winning_partner_exists),
    )?;
    d.set_item(
        "winning_partner_count",
        PyArray1::from_vec(py, labels.winning_partner_count),
    )?;
    Ok(d)
}

/// A Rust-collated ACT batch as the NumPy arrays `PackedACTBatch` accepts.
fn packed_act_to_dict<'py>(
    py: Python<'py>,
    packed: act_encoder::PackedActBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let positions = packed.position_count;
    let windows = packed.window_pattern_class.len();
    let legal = packed.legal_to_cell_index.len();
    let window_numeric_width = packed.window_numeric_width;
    let action_tactical_numeric_width = packed.action_tactical_numeric_width;
    let global_numeric_width = packed.global_numeric_width;
    let d = PyDict::new(py);
    d.set_item("position_count", positions)?;
    d.set_item("radius_orbit_bound", packed.radius_orbit_bound)?;
    d.set_item("cell_offsets", PyArray1::from_vec(py, packed.cell_offsets))?;
    d.set_item(
        "window_offsets",
        PyArray1::from_vec(py, packed.window_offsets),
    )?;
    d.set_item(
        "legal_offsets",
        PyArray1::from_vec(py, packed.legal_offsets),
    )?;
    d.set_item(
        "adjacency_offsets",
        PyArray1::from_vec(py, packed.adjacency_offsets),
    )?;
    d.set_item(
        "radius_offsets",
        PyArray1::from_vec(py, packed.radius_offsets),
    )?;
    d.set_item(
        "cell_occupancy",
        PyArray1::from_vec(py, packed.cell_occupancy),
    )?;
    d.set_item(
        "cell_is_legal",
        PyArray1::from_vec(py, packed.cell_is_legal),
    )?;
    d.set_item(
        "cell_nearest_bucket",
        PyArray1::from_vec(py, packed.cell_nearest_bucket),
    )?;
    d.set_item(
        "legal_to_cell_index",
        PyArray1::from_vec(py, packed.legal_to_cell_index),
    )?;
    d.set_item(
        "window_id",
        PyArray1::from_vec(py, packed.window_id).reshape([windows, 3])?,
    )?;
    d.set_item(
        "window_pattern_class",
        PyArray1::from_vec(py, packed.window_pattern_class),
    )?;
    d.set_item(
        "window_status",
        PyArray1::from_vec(py, packed.window_status),
    )?;
    d.set_item("window_axis", PyArray1::from_vec(py, packed.window_axis))?;
    d.set_item(
        "window_numeric",
        PyArray1::from_vec(py, packed.window_numeric).reshape([windows, window_numeric_width])?,
    )?;
    d.set_item(
        "window_cell_index",
        PyArray1::from_vec(py, packed.window_cell_index).reshape([windows, 6])?,
    )?;
    d.set_item(
        "window_incidence_class",
        PyArray1::from_vec(py, packed.window_incidence_class).reshape([windows, 6])?,
    )?;
    d.set_item(
        "window_incidence_mask",
        PyArray1::from_vec(py, packed.window_incidence_mask).reshape([windows, 6])?,
    )?;
    d.set_item(
        "adjacency_src",
        PyArray1::from_vec(py, packed.adjacency_src),
    )?;
    d.set_item(
        "adjacency_dst",
        PyArray1::from_vec(py, packed.adjacency_dst),
    )?;
    d.set_item(
        "adjacency_axis",
        PyArray1::from_vec(py, packed.adjacency_axis),
    )?;
    d.set_item("radius_src", PyArray1::from_vec(py, packed.radius_src))?;
    d.set_item("radius_dst", PyArray1::from_vec(py, packed.radius_dst))?;
    d.set_item("radius_orbit", PyArray1::from_vec(py, packed.radius_orbit))?;
    d.set_item(
        "radius_axis_or_neg1",
        PyArray1::from_vec(py, packed.radius_axis_or_neg1),
    )?;
    d.set_item(
        "action_window_index",
        PyArray1::from_vec(py, packed.action_window_index).reshape([legal, 3, 6])?,
    )?;
    d.set_item(
        "action_post1_class",
        PyArray1::from_vec(py, packed.action_post1_class).reshape([legal, 3, 6])?,
    )?;
    d.set_item(
        "action_pre_status",
        PyArray1::from_vec(py, packed.action_pre_status).reshape([legal, 3, 6])?,
    )?;
    d.set_item(
        "action_tactical_numeric",
        PyArray1::from_vec(py, packed.action_tactical_numeric)
            .reshape([legal, action_tactical_numeric_width])?,
    )?;
    d.set_item("phase_id", PyArray1::from_vec(py, packed.phase_id))?;
    d.set_item(
        "moves_remaining",
        PyArray1::from_vec(py, packed.moves_remaining),
    )?;
    d.set_item(
        "global_numeric",
        PyArray1::from_vec(py, packed.global_numeric).reshape([positions, global_numeric_width])?,
    )?;
    Ok(d)
}

/// Build a collated MantisNet batch from positions, in parallel.
///
/// The production twin of `mantisnet.builder`'s Python path, held equal to it
/// field for field by that package's parity tests. Raises `ValueError` on a
/// terminal position.
#[pyfunction]
fn build_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyRef<'py, Position>>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned: Vec<engine::Position> = positions.iter().map(|p| p.inner.clone()).collect();
    let raw = py
        .detach(|| encoder::build_batch(&owned))
        .map_err(PyValueError::new_err)?;
    raw_to_dict(py, raw)
}

/// Replay each game's first `ts[i]` placements and build the batch, in
/// parallel — the fitting path, where a stored position is a move prefix.
#[pyfunction]
fn build_batch_prefixes<'py>(
    py: Python<'py>,
    games: Vec<Vec<(i16, i16)>>,
    ts: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let raw = py
        .detach(|| encoder::build_batch_prefixes(&games, &ts))
        .map_err(PyValueError::new_err)?;
    raw_to_dict(py, raw)
}

/// Build one position-local MantisNet-ACT graph.
#[pyfunction]
fn build_act_graph<'py>(
    py: Python<'py>,
    position: PyRef<'py, Position>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_act_config(config)?;
    let owned = position.inner.clone();
    let graph = py
        .detach(|| act_encoder::build(&owned, &config))
        .map_err(PyValueError::new_err)?;
    act_graph_to_dict(py, graph)
}

/// Compute section 24.1's action labels from one engine position.
#[pyfunction]
fn build_act_aux_labels<'py>(
    py: Python<'py>,
    position: PyRef<'py, Position>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_act_config(config)?;
    let owned = position.inner.clone();
    let labels = py
        .detach(|| act_encoder::build_aux_labels(&owned, &config))
        .map_err(PyValueError::new_err)?;
    act_aux_labels_to_dict(py, labels)
}

/// Build and collate MantisNet-ACT graphs in parallel.
#[pyfunction]
fn build_act_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyRef<'py, Position>>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_act_config(config)?;
    let owned: Vec<engine::Position> = positions.iter().map(|p| p.inner.clone()).collect();
    let packed = py
        .detach(|| act_encoder::build_packed_batch(&owned, &config))
        .map_err(PyValueError::new_err)?;
    packed_act_to_dict(py, packed)
}

/// Replay move prefixes and build and collate MantisNet-ACT graphs in parallel.
#[pyfunction]
fn build_act_batch_prefixes<'py>(
    py: Python<'py>,
    games: Vec<Vec<(i16, i16)>>,
    ts: Vec<usize>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let config = parse_act_config(config)?;
    let packed = py
        .detach(|| act_encoder::build_packed_batch_prefixes(&games, &ts, &config))
        .map_err(PyValueError::new_err)?;
    packed_act_to_dict(py, packed)
}

/// The module: `Position`, the batch builders, and the version constants a
/// checkpoint pins.
#[pymodule]
fn hexo_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Position>()?;
    m.add_function(wrap_pyfunction!(build_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_batch_prefixes, m)?)?;
    m.add_function(wrap_pyfunction!(build_act_graph, m)?)?;
    m.add_function(wrap_pyfunction!(build_act_aux_labels, m)?)?;
    m.add_function(wrap_pyfunction!(build_act_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_act_batch_prefixes, m)?)?;
    m.add("RULES_VERSION", engine::RULES_VERSION)?;
    m.add("ACTION_ORDER_VERSION", engine::ACTION_ORDER_VERSION)?;
    m.add("LEGAL_RADIUS", engine::LEGAL_RADIUS)?;
    m.add("MODEL_REPR_VERSION", MODEL_REPR_VERSION)?;
    // A host orchestrator opens every seat with the three versions of
    // `CONTAINER_SPEC.md` §3.1's handshake. Two of them are the engine's and
    // already here; the third is the runner's, and a Python orchestrator
    // holding its own copy of that number is the drift this re-export exists
    // to prevent.
    m.add("PROTOCOL_VERSION", hexo_runner::PROTOCOL_VERSION)?;
    Ok(())
}
