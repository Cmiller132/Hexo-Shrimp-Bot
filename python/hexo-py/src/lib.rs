//! Python bindings for `hexo-engine`: the read surface a model builder needs,
//! and nothing that could bypass the rules.
//!
//! The surface is `MODEL_SPEC.md` §11's input list — stones, legal moves in
//! canonical order, `moves_remaining` — plus `windows_through`, which exists so
//! a builder test can check window enumeration against the engine as an
//! independent oracle (§12.1). Positions are created empty or by replay, never
//! deserialised: a board-shaped constructor would be a rule-bypass hole, which
//! is the same argument `ENGINE_SPEC.md` §12 makes for the engine itself.

use hexo_engine as engine;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

    /// The 18 win windows through `(q, r)`: `(axis, start_q, start_r, mask_p0,
    /// mask_p1)`, where bit `k` of a mask is the cell `k` steps from the start
    /// along the axis. Axes are `0 = Q (1,0)`, `1 = R (0,1)`, `2 = QR (1,-1)`.
    ///
    /// This is the engine's own window walk, exposed for the builder-oracle
    /// test. The builder must not call it (`MODEL_SPEC.md` §12.1).
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

/// The module: `Position` plus the version constants a checkpoint pins.
#[pymodule]
fn hexo_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Position>()?;
    m.add("RULES_VERSION", engine::RULES_VERSION)?;
    m.add("ACTION_ORDER_VERSION", engine::ACTION_ORDER_VERSION)?;
    m.add("LEGAL_RADIUS", engine::LEGAL_RADIUS)?;
    Ok(())
}
