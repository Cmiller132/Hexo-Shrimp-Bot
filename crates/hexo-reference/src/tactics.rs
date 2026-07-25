//! Incremental six-cell window tracking.
//!
//! A placement touches exactly 18 length-6 windows: 3 axes times 6 possible
//! offsets inside the window. The store keeps those windows incrementally so
//! wins and threats can be read from compact six-bit masks.

//! VENDORED CHANGE: the `ahash` hash map swapped for the `std::collections`
//! one, `serde` derives removed, and the unit tests plus the test-only
//! `update_for_placement` wrapper removed.

use super::coord::{hex_distance, HexCoord};
use super::state::Player;
use std::collections::HashMap;
use std::{fmt, ops::Deref};

/// Number of cells in a win/threat window.
pub const WINDOW_LEN: i16 = 6;

/// Number of six-cell windows affected by one placement.
pub const WINDOWS_PER_PLACEMENT: usize = 3 * WINDOW_LEN as usize;

const WINDOW_MASK: u8 = 0b0011_1111;

/// One of the three unique straight-line axes on the hex grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Increasing q: `(1, 0)`.
    Q,
    /// Increasing r: `(0, 1)`.
    R,
    /// Increasing q while decreasing r: `(1, -1)`.
    QR,
}

impl Axis {
    /// All unique axes. Opposite directions are represented by different starts.
    pub const ALL: [Self; 3] = [Self::Q, Self::R, Self::QR];

    /// Stable order for sorting/debug output.
    pub const fn index(self) -> u8 {
        match self {
            Self::Q => 0,
            Self::R => 1,
            Self::QR => 2,
        }
    }

    /// Direction vector for walking this axis.
    pub const fn vector(self) -> HexCoord {
        match self {
            Self::Q => HexCoord { q: 1, r: 0 },
            Self::R => HexCoord { q: 0, r: 1 },
            Self::QR => HexCoord { q: 1, r: -1 },
        }
    }
}

/// Canonical identity of one length-6 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowKey {
    /// First coordinate in the window.
    pub start: HexCoord,
    /// Axis along which the six cells are read.
    pub axis: Axis,
}

impl WindowKey {
    /// Coordinate at position `index` in this window.
    pub fn coord_at(self, index: u8) -> HexCoord {
        self.start + self.axis.vector().scale(index as i16)
    }

    /// All six coordinates in this window.
    pub fn cells(self) -> [HexCoord; WINDOW_LEN as usize] {
        let mut cells = [HexCoord::ZERO; WINDOW_LEN as usize];
        for index in 0..WINDOW_LEN as u8 {
            cells[index as usize] = self.coord_at(index);
        }
        cells
    }

    /// True when `coord` is one of this window's six cells.
    pub fn contains(self, coord: HexCoord) -> bool {
        self.cells().contains(&coord)
    }

    /// True when two windows share at least one cell.
    pub fn intersects(self, other: Self) -> bool {
        self.cells().iter().any(|cell| other.contains(*cell))
    }

    /// True when the windows do not overlap but have adjacent cells.
    pub fn touches(self, other: Self) -> bool {
        if self.intersects(other) {
            return false;
        }

        let left_cells = self.cells();
        let right_cells = other.cells();
        left_cells.iter().any(|left| {
            right_cells
                .iter()
                .any(|right| hex_distance(*left, *right) == 1)
        })
    }

    /// True when two windows either overlap or have adjacent cells.
    pub fn intersects_or_touches(self, other: Self) -> bool {
        self.intersects(other) || self.touches(other)
    }
}

/// Read view of one length-6 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowEntry {
    key: WindowKey,
    masks: [u8; 2],
}

impl WindowEntry {
    /// Canonical start and axis.
    pub fn key(self) -> WindowKey {
        self.key
    }

    /// All six coordinates in this window.
    pub fn cells(self) -> [HexCoord; WINDOW_LEN as usize] {
        self.key.cells()
    }

    /// Mask for one player's stones.
    pub fn mask(self, player: Player) -> u8 {
        self.masks[player.index()]
    }

    /// Number of stones the player has in this window.
    pub fn count(self, player: Player) -> u8 {
        self.mask(player).count_ones() as u8
    }

    /// All occupied cells, regardless of owner.
    pub fn occupied_mask(self) -> u8 {
        self.masks[Player::Player0.index()] | self.masks[Player::Player1.index()]
    }

    /// Empty positions inside the six-cell window.
    pub fn empty_mask(self) -> u8 {
        !self.occupied_mask() & WINDOW_MASK
    }

    /// Coordinates occupied by `player` inside this window.
    pub fn stone_cells(self, player: Player) -> Vec<HexCoord> {
        self.cells_for_mask(self.mask(player))
    }

    /// Empty coordinates inside this window.
    pub fn empty_cells(self) -> Vec<HexCoord> {
        self.cells_for_mask(self.empty_mask())
    }

    /// All occupied coordinates with their owning players.
    pub fn occupied_cells(self) -> Vec<(HexCoord, Player)> {
        let mut cells = Vec::with_capacity(self.occupied_mask().count_ones() as usize);
        for player in [Player::Player0, Player::Player1] {
            cells.extend(
                self.stone_cells(player)
                    .into_iter()
                    .map(|coord| (coord, player)),
            );
        }
        cells
    }

    /// Player who owns this active window, if it is active.
    pub fn active_player(self) -> Option<Player> {
        match (
            self.masks[Player::Player0.index()] != 0,
            self.masks[Player::Player1.index()] != 0,
        ) {
            (true, false) => Some(Player::Player0),
            (false, true) => Some(Player::Player1),
            _ => None,
        }
    }

    /// True when the window contains stones from exactly one player.
    pub fn is_active(self) -> bool {
        self.active_player().is_some()
    }

    /// Player who owns this threat, if the window is currently a threat.
    pub fn threat_player(self) -> Option<Player> {
        let player = self.active_player()?;
        (self.count(player) >= 4).then_some(player)
    }

    /// True when this window has at least four stones from one player and none
    /// from the other.
    pub fn is_threat(self) -> bool {
        self.threat_player().is_some()
    }

    /// True when this active window has at least four stones for `player`.
    pub fn is_threat_for(self, player: Player) -> bool {
        self.threat_player() == Some(player)
    }

    /// True when this active window is completely filled by `player`.
    pub fn is_win_for(self, player: Player) -> bool {
        self.active_player() == Some(player) && self.count(player) == WINDOW_LEN as u8
    }

    /// True when this window shares at least one cell with `other`.
    pub fn intersects(self, other: Self) -> bool {
        self.key.intersects(other.key)
    }

    /// True when this window does not overlap `other` but has adjacent cells.
    pub fn touches(self, other: Self) -> bool {
        self.key.touches(other.key)
    }

    /// True when this window either overlaps `other` or has adjacent cells.
    pub fn intersects_or_touches(self, other: Self) -> bool {
        self.key.intersects_or_touches(other.key)
    }

    fn cells_for_mask(self, mask: u8) -> Vec<HexCoord> {
        let mut cells = Vec::with_capacity(mask.count_ones() as usize);
        for index in 0..WINDOW_LEN as u8 {
            if mask & (1u8 << index) != 0 {
                cells.push(self.key.coord_at(index));
            }
        }
        cells
    }
}

const EMPTY_WINDOW_KEY: WindowKey = WindowKey {
    start: HexCoord::ZERO,
    axis: Axis::Q,
};

/// Fixed-capacity list of window keys affected by one placement.
///
/// A placement can affect at most 18 windows, so this keeps `WindowUpdate`
/// stack-backed while still exposing slice-like read access.
#[derive(Clone, Eq)]
pub struct WindowKeyList {
    keys: [WindowKey; WINDOWS_PER_PLACEMENT],
    len: u8,
}

impl WindowKeyList {
    /// Create an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// True when no keys are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stored keys as a slice.
    pub fn as_slice(&self) -> &[WindowKey] {
        &self.keys[..self.len()]
    }

    fn push(&mut self, key: WindowKey) {
        assert!(
            self.len() < WINDOWS_PER_PLACEMENT,
            "window key list capacity exceeded"
        );
        let index = self.len();
        self.keys[index] = key;
        self.len += 1;
    }
}

impl Default for WindowKeyList {
    fn default() -> Self {
        Self {
            keys: [EMPTY_WINDOW_KEY; WINDOWS_PER_PLACEMENT],
            len: 0,
        }
    }
}

impl fmt::Debug for WindowKeyList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl PartialEq for WindowKeyList {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl AsRef<[WindowKey]> for WindowKeyList {
    fn as_ref(&self) -> &[WindowKey] {
        self.as_slice()
    }
}

impl Deref for WindowKeyList {
    type Target = [WindowKey];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Incremental result produced by one placement's window updates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowUpdate {
    /// All windows touched by the placement.
    pub changed: WindowKeyList,
    /// Changed windows that are now threats for the placed player.
    pub threats: WindowKeyList,
    /// Changed windows that are now wins for the placed player.
    pub winning_windows: WindowKeyList,
}

impl WindowUpdate {
    /// True if this placement completed a six-in-line window.
    pub fn has_win(&self) -> bool {
        !self.winning_windows.is_empty()
    }

    /// True if this placement created or preserved a threat.
    pub fn has_threat(&self) -> bool {
        !self.threats.is_empty()
    }
}

/// Maintained index of all touched windows.
#[derive(Clone, Debug, Default)]
pub struct WindowStore {
    masks_by_key: HashMap<WindowKey, [u8; 2]>,
    /// Incrementally maintained set of currently-active threat windows
    /// (single-colour, count >= 4) keyed by window -> owning player. Updated in
    /// lockstep with `masks_by_key` in `update_for_placement_with_delta` /
    /// `restore_delta` (only the <= 18 windows a placement touches can change
    /// threat status, so this is O(18) per placement). It is an exact mirror of
    /// `entries().filter_map(threat_player)` and changes NO public output — it
    /// exists only so `has_threats()` is O(1), letting the hexgt TSS hot path
    /// skip its full `threats()` scan in the (common) threat-free case.
    live_threats: HashMap<WindowKey, Player>,
}

/// Incremental window-mask changes made by one placement.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowStoreDelta {
    previous_masks: Vec<(WindowKey, Option<[u8; 2]>)>,
}

impl WindowStore {
    /// Create an empty window store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of known/touched windows.
    pub fn len(&self) -> usize {
        self.masks_by_key.len()
    }

    /// True when no windows have been touched yet.
    pub fn is_empty(&self) -> bool {
        self.masks_by_key.is_empty()
    }

    /// Fetch a window entry by canonical key.
    pub fn entry(&self, key: WindowKey) -> Option<WindowEntry> {
        self.masks_by_key
            .get(&key)
            .copied()
            .map(|masks| WindowEntry { key, masks })
    }

    /// Iterate all known windows.
    pub fn entries(&self) -> impl Iterator<Item = WindowEntry> + '_ {
        self.masks_by_key.iter().map(|(key, masks)| WindowEntry {
            key: *key,
            masks: *masks,
        })
    }

    /// Current threat windows for one player.
    pub fn threat_entries(&self, player: Player) -> impl Iterator<Item = WindowEntry> + '_ {
        self.entries()
            .filter(move |entry| (*entry).is_threat_for(player))
    }

    /// Current threat windows for both players.
    pub fn threats(&self) -> impl Iterator<Item = (Player, WindowEntry)> + '_ {
        self.entries()
            .filter_map(|entry| entry.threat_player().map(|player| (player, entry)))
    }

    /// True when at least one active >= 4 (single-colour) threat window exists for
    /// either player. O(1): reads the incrementally maintained `live_threats`
    /// index. Exactly equivalent to `threats().next().is_some()` (asserted by the
    /// `live_threats_mirror_*` tests over random games with undo).
    pub fn has_threats(&self) -> bool {
        !self.live_threats.is_empty()
    }

    /// Build a window store from a flat placement list, without a full
    /// `HexoState`.
    ///
    /// Replays each `(coord, player)` through the same incremental update the
    /// engine uses during play, so the resulting store is bit-identical to one
    /// produced by `Board::place` over the same sequence (same `masks_by_key`
    /// and `live_threats`). The read API (`entry`/`entries`/`threats`/
    /// `has_threats`) is therefore identical to a played-out store.
    ///
    /// Intended for the train-time feature-expand kernel, which owns raw
    /// placement facts (history rows) but no `HexoState`, so it cannot reach the
    /// board's incremental store. Placements must be pairwise-distinct cells (as
    /// any real game history is); a repeated cell trips the same
    /// double-occupancy `debug_assert` as normal play.
    pub fn from_placements(placements: &[(HexCoord, Player)]) -> Self {
        let mut store = Self::new();
        for &(coord, player) in placements {
            store.update_for_placement_with_delta(coord, player);
        }
        store
    }

    /// Update affected windows and return the masks needed for undo.
    pub(crate) fn update_for_placement_with_delta(
        &mut self,
        coord: HexCoord,
        player: Player,
    ) -> (WindowUpdate, WindowStoreDelta) {
        let mut update = WindowUpdate::default();
        let mut delta = WindowStoreDelta::default();

        for axis in Axis::ALL {
            for offset in 0..WINDOW_LEN as u8 {
                let (key, bit) = window_containing(coord, axis, offset);
                delta
                    .previous_masks
                    .push((key, self.masks_by_key.get(&key).copied()));
                let masks = self.masks_by_key.entry(key).or_insert([0; 2]);
                debug_assert_eq!((masks[0] | masks[1]) & bit, 0);
                masks[player.index()] |= bit;

                let entry = WindowEntry { key, masks: *masks };
                update.changed.push(key);

                if entry.is_threat_for(player) {
                    update.threats.push(key);
                }
                if entry.is_win_for(player) {
                    update.winning_windows.push(key);
                }

                // Keep the incremental threat index exact: this window's threat
                // status is fully determined by its new mask, so set-or-clear
                // unconditionally (idempotent; correct regardless of prior state).
                match entry.threat_player() {
                    Some(owner) => {
                        self.live_threats.insert(key, owner);
                    }
                    None => {
                        self.live_threats.remove(&key);
                    }
                }
            }
        }

        (update, delta)
    }

    pub(crate) fn restore_delta(&mut self, delta: WindowStoreDelta) {
        for (key, previous) in delta.previous_masks.into_iter().rev() {
            if let Some(masks) = previous {
                self.masks_by_key.insert(key, masks);
                // Restore the threat index from the restored mask (set-or-clear,
                // mirroring the forward update above).
                match (WindowEntry { key, masks }).threat_player() {
                    Some(owner) => {
                        self.live_threats.insert(key, owner);
                    }
                    None => {
                        self.live_threats.remove(&key);
                    }
                }
            } else {
                self.masks_by_key.remove(&key);
                self.live_threats.remove(&key);
            }
        }
    }
}

fn window_containing(coord: HexCoord, axis: Axis, offset: u8) -> (WindowKey, u8) {
    let start = coord - axis.vector().scale(offset as i16);
    (WindowKey { start, axis }, 1u8 << offset)
}
