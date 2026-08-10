//! Linear-time execution-plan construction for packed MantisNet-ACT batches.
//!
//! The Python model consumes several stable CSR views of the packed graph.
//! Building those views with general-purpose sorts was substantially more
//! expensive than building the graph itself.  This module uses bounded integer
//! keys and stable counting scatter instead.  Rows tied on a key retain their
//! packed order exactly, matching NumPy's stable `argsort` reference.

use crate::act_encoder::{self, ActBuilderConfig, D6RelationMode, PackedActBatch};
use rayon::join;
use rayon::prelude::*;
use std::sync::Mutex;

const WINDOW_LEN: usize = 6;
const INCIDENCE_RELATIONS: usize = 2_187;
const WINDOW_PATTERN_CLASSES: usize = 378;
const WINDOW_STATUSES: usize = 4;
const POST1_CLASSES: usize = 729;
const CLASS_BLOCK_ROWS: usize = 128;
const CELL_OCCUPANCY_CLASSES: usize = 3;
const CELL_LEGAL_CLASSES: usize = 2;
const CELL_NEAREST_CLASSES: usize = 10;
const ORBIT_RELATIONS: usize = 52;

/// One stable CSR ordering of an edge family.
///
/// `left` and `right` are the two non-key edge columns.  Their semantic names
/// are fixed by the containing family and view, and the PyO3 boundary exposes
/// descriptive flat names (for example `plan_incidence_dst_cell`).  `axis` is
/// populated for the always-built axis superset and empty for invariant-only
/// radius views.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StableEdgeView {
    /// CSR offsets for the view's grouping key.
    pub ptr: Vec<i32>,
    /// First non-key edge column in stable key-major order.
    pub left: Vec<i32>,
    /// Second non-key edge column in stable key-major order.
    pub right: Vec<i32>,
    /// Axis route in stable key-major order, or empty for an invariant-only view.
    pub axis: Vec<i32>,
}

/// Incidence rows and their window-, cell-, and relation-major views.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct IncidencePlanArrays {
    /// Represented cell for each canonical live incidence row.
    pub cell: Vec<i64>,
    /// Persistent window for each canonical live incidence row.
    pub window: Vec<i64>,
    /// Joint pattern/slot relation for each canonical live incidence row.
    pub relation: Vec<i64>,
    /// Native window axis for each canonical live incidence row.
    pub axis: Vec<i64>,
    /// Window-major view; payloads are cell and relation.
    pub dst: StableEdgeView,
    /// Cell-major view; payloads are window and relation.
    pub src: StableEdgeView,
    /// Relation-major view; payloads are cell and window.
    pub rel: StableEdgeView,
}

/// Adjacency's derived relation column and three stable message views.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AdjacencyPlanArrays {
    /// Constant distance-one relation for every adjacency row.
    pub relation: Vec<i64>,
    /// Destination-major view; payloads are source and relation.
    pub dst: StableEdgeView,
    /// Source-major view; payloads are destination and relation.
    pub src: StableEdgeView,
    /// Relation-major view; payloads are source and destination.
    pub rel: StableEdgeView,
}

/// Radius invariant views plus the materialised stable routed subset.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RadiusPlanArrays {
    /// Joint geometry/source-colour relation for every full radius row.
    pub relation: Vec<i64>,
    /// Full destination-major invariant view.
    pub dst: StableEdgeView,
    /// Full source-major invariant view.
    pub src: StableEdgeView,
    /// Full relation-major invariant view.
    pub rel: StableEdgeView,
    /// Full-family row index of every routed radius row.
    pub axis_rows: Vec<i64>,
    /// Source column of the stable routed subset.
    pub routed_src: Vec<i64>,
    /// Destination column of the stable routed subset.
    pub routed_dst: Vec<i64>,
    /// Joint relation column of the stable routed subset.
    pub routed_relation: Vec<i64>,
    /// Nonnegative axis column of the stable routed subset.
    pub routed_axis: Vec<i64>,
    /// Routed-subset destination-major axis view.
    pub axis_dst: StableEdgeView,
    /// Routed-subset source-major axis view.
    pub axis_src: StableEdgeView,
    /// Routed-subset relation-major axis view.
    pub axis_rel: StableEdgeView,
}

/// Stable class-major rows and the fixed 128-row reduction partition.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ClassRowPlanArrays {
    /// CSR offsets of class-major `rows` runs.
    pub ptr: Vec<i32>,
    /// Original row indices stably grouped by class.
    pub rows: Vec<i32>,
    /// CSR offsets of each class's block grid.
    pub block_ptr: Vec<i32>,
    /// Starting index in `rows` for each reduction block.
    pub block_starts: Vec<i32>,
    /// Length, at most 128, of each reduction block.
    pub block_lengths: Vec<i32>,
}

/// Stable action rows grouped by persistent source window.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SourceWindowPlanArrays {
    /// CSR offsets of persistent-window action-row runs.
    pub ptr: Vec<i32>,
    /// Live action rows stably grouped by source window.
    pub rows: Vec<i32>,
    /// Sentinel action rows in original row order.
    pub sentinel_rows: Vec<i32>,
}

/// Stable legal-action rows grouped by their base cell (including sentinel).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GatherRowPlanArrays {
    /// CSR offsets of base-cell source runs, including the sentinel source.
    pub ptr: Vec<i32>,
    /// Legal rows stably grouped by base-cell source.
    pub rows: Vec<i32>,
}

/// Multi-family ragged latent-row ownership.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LatentSegmentArrays {
    /// Flat `[positions, families, 2]`.
    pub ranges: Vec<i32>,
    /// Flat `[positions, families]`.
    pub range_base: Vec<i32>,
    /// Total latent input rows owned by each position.
    pub counts: Vec<i32>,
    /// Position owner for each family-major latent input row.
    pub row_pos: Vec<i64>,
}

/// Every additional array needed to assemble Python's `ACTPlans` containers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ActPlanArrays {
    /// Cell/window incidence execution views.
    pub incidence: IncidencePlanArrays,
    /// Optional cell-adjacency execution views, or an empty wire prefix.
    pub adjacency: AdjacencyPlanArrays,
    /// Optional occupied-radius execution views, or an empty wire prefix.
    pub radius: RadiusPlanArrays,

    /// Cell-occupancy embedding rows.
    pub class_cell_occupancy: ClassRowPlanArrays,
    /// Cell-legality embedding rows.
    pub class_cell_legal: ClassRowPlanArrays,
    /// Cell-nearest-bucket embedding rows.
    pub class_cell_nearest: ClassRowPlanArrays,
    /// Window-pattern embedding rows.
    pub class_window_pattern: ClassRowPlanArrays,
    /// Window-status embedding rows.
    pub class_window_status: ClassRowPlanArrays,
    /// Action post-placement-class embedding rows.
    pub class_action_post1: ClassRowPlanArrays,
    /// Action pre-status embedding rows.
    pub class_action_pre_status: ClassRowPlanArrays,

    /// Action rows grouped by persistent source window.
    pub action_source_window: SourceWindowPlanArrays,
    /// Legal rows grouped by base cell.
    pub action_base_cell: GatherRowPlanArrays,

    /// Position owner of every packed cell row.
    pub cell_row_pos: Vec<i64>,
    /// Position owner of every packed window row.
    pub window_row_pos: Vec<i64>,
    /// Position owner of every packed legal row.
    pub legal_row_pos: Vec<i64>,
    /// Phase id broadcast to packed cell rows.
    pub cell_phase: Vec<i64>,
    /// Phase id broadcast to packed window rows.
    pub window_phase: Vec<i64>,
    /// Phase id broadcast to packed action rows.
    pub action_phase: Vec<i64>,
    /// Cell/window latent segment layout.
    pub state_segment: LatentSegmentArrays,
    /// Legal-action latent segment layout.
    pub action_segment: LatentSegmentArrays,
}

/// A packed graph and its execution-plan wire arrays.
#[derive(Debug, PartialEq)]
pub struct PlannedActBatch {
    /// Original packed graph wire fields.
    pub packed: PackedActBatch,
    /// Derived execution-plan wire fields.
    pub plans: ActPlanArrays,
}

fn checked_i32(name: &str, value: usize) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{name} value {value} exceeds signed int32"))
}

fn checked_key(name: &str, value: i64, bound: usize, row: usize) -> Result<usize, String> {
    let key =
        usize::try_from(value).map_err(|_| format!("{name} row {row} has negative key {value}"))?;
    if key >= bound {
        return Err(format!(
            "{name} row {row} has key {value} outside 0..{bound}"
        ));
    }
    Ok(key)
}

fn csr_ptr(name: &str, counts: &[usize], rows: usize) -> Result<Vec<i32>, String> {
    checked_i32(name, rows)?;
    let mut ptr = Vec::with_capacity(counts.len() + 1);
    ptr.push(0);
    let mut total = 0usize;
    for &count in counts {
        total = total
            .checked_add(count)
            .ok_or_else(|| format!("{name} row count overflows usize"))?;
        ptr.push(checked_i32(name, total)?);
    }
    if total != rows {
        return Err(format!("{name} counts span {total} rows, expected {rows}"));
    }
    Ok(ptr)
}

/// Stable counting scatter of two payload columns and an optional axis.
fn stable_edge_view(
    name: &str,
    key: &[i64],
    left: &[i64],
    right: &[i64],
    axis: Option<&[i64]>,
    key_count: usize,
    already_sorted: bool,
) -> Result<StableEdgeView, String> {
    let rows = key.len();
    if left.len() != rows || right.len() != rows || axis.is_some_and(|values| values.len() != rows)
    {
        return Err(format!("{name} columns have different row counts"));
    }
    checked_i32(name, key_count)?;
    checked_i32(name, rows)?;

    let mut counts = vec![0usize; key_count];
    let mut previous = None;
    for (row, &value) in key.iter().enumerate() {
        let value = checked_key(name, value, key_count, row)?;
        if already_sorted && previous.is_some_and(|prior| value < prior) {
            return Err(format!(
                "{name} declares sorted keys but row {row} decreases"
            ));
        }
        previous = Some(value);
        counts[value] += 1;
    }
    let ptr = csr_ptr(name, &counts, rows)?;

    let mut out_left = vec![0i32; rows];
    let mut out_right = vec![0i32; rows];
    let mut out_axis = axis.map(|_| vec![0i32; rows]).unwrap_or_default();
    if already_sorted {
        for row in 0..rows {
            out_left[row] = i32::try_from(left[row])
                .map_err(|_| format!("{name}.left row {row} value {} exceeds int32", left[row]))?;
            out_right[row] = i32::try_from(right[row]).map_err(|_| {
                format!("{name}.right row {row} value {} exceeds int32", right[row])
            })?;
            if let Some(axis) = axis {
                out_axis[row] = i32::try_from(axis[row]).map_err(|_| {
                    format!("{name}.axis row {row} value {} exceeds int32", axis[row])
                })?;
            }
        }
    } else {
        let mut next: Vec<usize> = ptr[..key_count]
            .iter()
            .map(|&value| value as usize)
            .collect();
        // Scanning input rows in canonical order is what makes the counting
        // scatter identical to NumPy's stable argsort for ties.
        for row in 0..rows {
            let key = checked_key(name, key[row], key_count, row)?;
            let output = next[key];
            next[key] += 1;
            out_left[output] = i32::try_from(left[row])
                .map_err(|_| format!("{name}.left row {row} value {} exceeds int32", left[row]))?;
            out_right[output] = i32::try_from(right[row]).map_err(|_| {
                format!("{name}.right row {row} value {} exceeds int32", right[row])
            })?;
            if let Some(axis) = axis {
                out_axis[output] = i32::try_from(axis[row]).map_err(|_| {
                    format!("{name}.axis row {row} value {} exceeds int32", axis[row])
                })?;
            }
        }
    }
    Ok(StableEdgeView {
        ptr,
        left: out_left,
        right: out_right,
        axis: out_axis,
    })
}

/// Build a cell-keyed view in parallel over the disjoint cell band owned by
/// each packed position.  Each position also owns one contiguous output row
/// band, so workers can scatter directly into the final vectors without a
/// merge copy.
fn stable_position_cell_view(
    name: &str,
    row_offsets: &[usize],
    cell_offsets: &[i64],
    key: &[i64],
    left: &[i64],
    right: &[i64],
    axis: Option<&[i64]>,
) -> Result<StableEdgeView, String> {
    let rows = key.len();
    if left.len() != rows
        || right.len() != rows
        || axis.is_some_and(|values| values.len() != rows)
        || row_offsets.len() != cell_offsets.len()
        || row_offsets.first() != Some(&0)
        || row_offsets.last() != Some(&rows)
        || cell_offsets.first() != Some(&0)
    {
        return Err(format!("{name} columns or packed offsets disagree"));
    }
    checked_i32(name, rows)?;
    let cells = usize::try_from(*cell_offsets.last().expect("offsets are nonempty"))
        .map_err(|_| format!("{name} final cell offset is negative"))?;
    checked_i32(name, cells)?;
    for position in 0..row_offsets.len() - 1 {
        let row_start = row_offsets[position];
        let row_end = row_offsets[position + 1];
        if row_end < row_start || row_end > rows {
            return Err(format!(
                "{name} row offsets are invalid at position {position}"
            ));
        }
        let cell_start = usize::try_from(cell_offsets[position])
            .map_err(|_| format!("{name} cell offset at position {position} is negative"))?;
        let cell_end = usize::try_from(cell_offsets[position + 1])
            .map_err(|_| format!("{name} cell offset after position {position} is negative"))?;
        if cell_end < cell_start || cell_end > cells {
            return Err(format!(
                "{name} cell offsets are invalid at position {position}"
            ));
        }
    }
    let mut counts = vec![0usize; cells];
    let mut out_left = vec![0i32; rows];
    let mut out_right = vec![0i32; rows];
    let mut out_axis = axis.map(|_| vec![0i32; rows]).unwrap_or_default();
    let error = Mutex::new(None::<(usize, String)>);
    rayon::scope(|scope| {
        let mut counts_tail = counts.as_mut_slice();
        let mut left_tail = out_left.as_mut_slice();
        let mut right_tail = out_right.as_mut_slice();
        let mut axis_tail = out_axis.as_mut_slice();
        for position in 0..row_offsets.len() - 1 {
            let row_start = row_offsets[position];
            let row_end = row_offsets[position + 1];
            let local_rows = row_end - row_start;
            let cell_start = match usize::try_from(cell_offsets[position]) {
                Ok(value) => value,
                Err(_) => {
                    *error.lock().expect("cell-view error mutex poisoned") = Some((
                        row_start,
                        format!("{name} cell offset at position {position} is negative"),
                    ));
                    break;
                }
            };
            let cell_end = match usize::try_from(cell_offsets[position + 1]) {
                Ok(value) if value >= cell_start => value,
                _ => {
                    *error.lock().expect("cell-view error mutex poisoned") = Some((
                        row_start,
                        format!("{name} cell offsets decrease at position {position}"),
                    ));
                    break;
                }
            };
            let local_cells = cell_end - cell_start;
            let (local_counts, counts_rest) = counts_tail.split_at_mut(local_cells);
            let (local_left, left_rest) = left_tail.split_at_mut(local_rows);
            let (local_right, right_rest) = right_tail.split_at_mut(local_rows);
            let (local_axis, axis_rest) = if axis.is_some() {
                axis_tail.split_at_mut(local_rows)
            } else {
                (&mut [][..], axis_tail)
            };
            let error = &error;
            scope.spawn(move |_| {
                let result = (|| {
                    for (local_row, row) in (row_start..row_end).enumerate() {
                        let global_key = checked_key(name, key[row], cell_end, row)?;
                        if global_key < cell_start {
                            return Err(format!(
                                "{name} row {row} crosses packed position {position}"
                            ));
                        }
                        local_counts[global_key - cell_start] += 1;
                        i32::try_from(left[row]).map_err(|_| {
                            format!(
                                "{name}.left row {row} value {} exceeds int32",
                                left[row]
                            )
                        })?;
                        i32::try_from(right[row]).map_err(|_| {
                            format!(
                                "{name}.right row {row} value {} exceeds int32",
                                right[row]
                            )
                        })?;
                        if let Some(axis) = axis {
                            i32::try_from(axis[row]).map_err(|_| {
                                format!(
                                    "{name}.axis row {row} value {} exceeds int32",
                                    axis[row]
                                )
                            })?;
                        }
                        debug_assert!(local_row < local_rows);
                    }
                    let mut next = Vec::with_capacity(local_cells);
                    let mut total = 0usize;
                    for &count in local_counts.iter() {
                        next.push(total);
                        total += count;
                    }
                    if total != local_rows {
                        return Err(format!(
                            "{name} position {position} counts span {total} rows, expected {local_rows}"
                        ));
                    }
                    for row in row_start..row_end {
                        let local_key = key[row] as usize - cell_start;
                        let output = next[local_key];
                        next[local_key] += 1;
                        local_left[output] = left[row] as i32;
                        local_right[output] = right[row] as i32;
                        if let Some(axis) = axis {
                            local_axis[output] = axis[row] as i32;
                        }
                    }
                    Ok::<(), String>(())
                })();
                if let Err(message) = result {
                    let mut first = error.lock().expect("cell-view error mutex poisoned");
                    if first.as_ref().is_none_or(|current| row_start < current.0) {
                        *first = Some((row_start, message));
                    }
                }
            });
            counts_tail = counts_rest;
            left_tail = left_rest;
            right_tail = right_rest;
            axis_tail = axis_rest;
        }
    });
    if let Some((_, message)) = error.into_inner().expect("cell-view error mutex poisoned") {
        return Err(message);
    }
    Ok(StableEdgeView {
        ptr: csr_ptr(name, &counts, rows)?,
        left: out_left,
        right: out_right,
        axis: out_axis,
    })
}

struct PositionRelationRows {
    ptr: Vec<usize>,
    rows: Vec<u32>,
}

/// Build a relation-keyed view without a serial global scatter.  Position
/// workers first produce compact stable row-id buckets.  Relation workers then
/// own disjoint final CSR runs and gather payloads directly into those runs.
fn stable_position_relation_view(
    name: &str,
    row_offsets: &[usize],
    key: &[i64],
    left: &[i64],
    right: &[i64],
    axis: Option<&[i64]>,
    relation_count: usize,
) -> Result<StableEdgeView, String> {
    let rows = key.len();
    if left.len() != rows
        || right.len() != rows
        || axis.is_some_and(|values| values.len() != rows)
        || row_offsets.first() != Some(&0)
        || row_offsets.last() != Some(&rows)
    {
        return Err(format!("{name} columns or packed offsets disagree"));
    }
    checked_i32(name, rows)?;
    checked_i32(name, relation_count)?;
    let positions = (0..row_offsets.len() - 1)
        .into_par_iter()
        .map(|position| {
            let start = row_offsets[position];
            let end = row_offsets[position + 1];
            let mut counts = vec![0usize; relation_count];
            for row in start..end {
                let relation = checked_key(name, key[row], relation_count, row)?;
                counts[relation] += 1;
                i32::try_from(left[row]).map_err(|_| {
                    format!("{name}.left row {row} value {} exceeds int32", left[row])
                })?;
                i32::try_from(right[row]).map_err(|_| {
                    format!("{name}.right row {row} value {} exceeds int32", right[row])
                })?;
                if let Some(axis) = axis {
                    i32::try_from(axis[row]).map_err(|_| {
                        format!("{name}.axis row {row} value {} exceeds int32", axis[row])
                    })?;
                }
            }
            let mut ptr = Vec::with_capacity(relation_count + 1);
            ptr.push(0usize);
            for count in counts {
                ptr.push(ptr.last().expect("relation ptr has a leading zero") + count);
            }
            let mut next = ptr[..relation_count].to_vec();
            let mut ordered_rows = vec![0u32; end - start];
            for (row, &relation) in key.iter().enumerate().take(end).skip(start) {
                let relation = relation as usize;
                let output = next[relation];
                next[relation] += 1;
                ordered_rows[output] = row as u32;
            }
            Ok::<_, String>(PositionRelationRows {
                ptr,
                rows: ordered_rows,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut counts = vec![0usize; relation_count];
    for position in &positions {
        for (relation, count) in counts.iter_mut().enumerate() {
            *count += position.ptr[relation + 1] - position.ptr[relation];
        }
    }
    let ptr = csr_ptr(name, &counts, rows)?;
    let mut out_left = vec![0i32; rows];
    let mut out_right = vec![0i32; rows];
    let mut out_axis = axis.map(|_| vec![0i32; rows]).unwrap_or_default();
    rayon::scope(|scope| {
        let mut left_tail = out_left.as_mut_slice();
        let mut right_tail = out_right.as_mut_slice();
        let mut axis_tail = out_axis.as_mut_slice();
        for (relation, &count) in counts.iter().enumerate() {
            let (relation_left, left_rest) = left_tail.split_at_mut(count);
            let (relation_right, right_rest) = right_tail.split_at_mut(count);
            let (relation_axis, axis_rest) = if axis.is_some() {
                axis_tail.split_at_mut(count)
            } else {
                (&mut [][..], axis_tail)
            };
            let positions = &positions;
            scope.spawn(move |_| {
                let mut output = 0usize;
                for position in positions {
                    let start = position.ptr[relation];
                    let end = position.ptr[relation + 1];
                    for &row in &position.rows[start..end] {
                        let row = row as usize;
                        relation_left[output] = left[row] as i32;
                        relation_right[output] = right[row] as i32;
                        if let Some(axis) = axis {
                            relation_axis[output] = axis[row] as i32;
                        }
                        output += 1;
                    }
                }
                debug_assert_eq!(output, count);
            });
            left_tail = left_rest;
            right_tail = right_rest;
            axis_tail = axis_rest;
        }
    });
    Ok(StableEdgeView {
        ptr,
        left: out_left,
        right: out_right,
        axis: out_axis,
    })
}

fn class_rows(
    name: &str,
    values: &[i64],
    class_count: usize,
) -> Result<ClassRowPlanArrays, String> {
    checked_i32(name, class_count)?;
    checked_i32(name, values.len())?;
    let mut counts = vec![0usize; class_count];
    for (row, &value) in values.iter().enumerate() {
        let class = checked_key(name, value, class_count, row)?;
        counts[class] += 1;
    }
    let ptr = csr_ptr(name, &counts, values.len())?;
    let mut next: Vec<usize> = ptr[..class_count]
        .iter()
        .map(|&value| value as usize)
        .collect();
    let mut rows = vec![0i32; values.len()];
    for (row, &class) in values.iter().enumerate() {
        let class = class as usize;
        let output = next[class];
        next[class] += 1;
        rows[output] = checked_i32(name, row)?;
    }

    let mut block_ptr = Vec::with_capacity(class_count + 1);
    let mut block_starts = Vec::new();
    let mut block_lengths = Vec::new();
    block_ptr.push(0);
    for class in 0..class_count {
        let start = ptr[class] as usize;
        let end = ptr[class + 1] as usize;
        for block_start in (start..end).step_by(CLASS_BLOCK_ROWS) {
            block_starts.push(checked_i32(name, block_start)?);
            block_lengths.push(checked_i32(name, CLASS_BLOCK_ROWS.min(end - block_start))?);
        }
        block_ptr.push(checked_i32(name, block_starts.len())?);
    }
    Ok(ClassRowPlanArrays {
        ptr,
        rows,
        block_ptr,
        block_starts,
        block_lengths,
    })
}

fn incidence_plans(batch: &PackedActBatch) -> Result<IncidencePlanArrays, String> {
    let windows = batch.window_pattern_class.len();
    let cells = batch.cell_occupancy.len();
    let slots = windows
        .checked_mul(WINDOW_LEN)
        .ok_or_else(|| "incidence slot count overflows usize".to_owned())?;
    if batch.window_incidence_mask.len() != slots
        || batch.window_cell_index.len() != slots
        || batch.window_incidence_class.len() != slots
        || batch.window_axis.len() != windows
    {
        return Err("packed incidence columns disagree with the window count".to_owned());
    }
    let rows = batch
        .window_incidence_mask
        .iter()
        .filter(|&&live| live)
        .count();
    let mut cell = Vec::with_capacity(rows);
    let mut window = Vec::with_capacity(rows);
    let mut relation = Vec::with_capacity(rows);
    let mut axis = Vec::with_capacity(rows);
    for row in 0..slots {
        if batch.window_incidence_mask[row] {
            let window_row = row / WINDOW_LEN;
            cell.push(batch.window_cell_index[row]);
            window.push(
                i64::try_from(window_row)
                    .map_err(|_| format!("incidence window row {window_row} exceeds int64"))?,
            );
            relation.push(batch.window_incidence_class[row]);
            axis.push(batch.window_axis[window_row]);
        }
    }
    let ((dst, src), rel) = join(
        || {
            join(
                || {
                    stable_edge_view(
                        "incidence destination",
                        &window,
                        &cell,
                        &relation,
                        Some(&axis),
                        windows,
                        true,
                    )
                },
                || {
                    stable_edge_view(
                        "incidence source",
                        &cell,
                        &window,
                        &relation,
                        Some(&axis),
                        cells,
                        false,
                    )
                },
            )
        },
        || {
            stable_edge_view(
                "incidence relation",
                &relation,
                &cell,
                &window,
                Some(&axis),
                INCIDENCE_RELATIONS,
                false,
            )
        },
    );
    Ok(IncidencePlanArrays {
        cell,
        window,
        relation,
        axis,
        dst: dst?,
        src: src?,
        rel: rel?,
    })
}

fn relation_vocabulary_size(cfg: &ActBuilderConfig) -> Result<usize, String> {
    match cfg.d6_relation_mode {
        D6RelationMode::Orbit48 => Ok(ORBIT_RELATIONS),
        D6RelationMode::CoarseDistanceAxis => cfg
            .d_max
            .checked_mul(2)
            .ok_or_else(|| "coarse relation vocabulary size overflows usize".to_owned()),
    }
}

fn adjacency_plans(
    batch: &PackedActBatch,
    cfg: &ActBuilderConfig,
) -> Result<AdjacencyPlanArrays, String> {
    if !cfg.use_cell_adjacency {
        return Ok(AdjacencyPlanArrays::default());
    }
    let rows = batch.adjacency_src.len();
    if batch.adjacency_dst.len() != rows || batch.adjacency_axis.len() != rows {
        return Err("packed adjacency columns have different row counts".to_owned());
    }
    let cells = batch.cell_occupancy.len();
    let relation_count = relation_vocabulary_size(cfg)?;
    // Distance one is the first exact orbit and the first coarse bucket, on-axis.
    let relation = vec![0i64; rows];
    let ((dst, src), rel) = join(
        || {
            join(
                || {
                    stable_edge_view(
                        "adjacency destination",
                        &batch.adjacency_dst,
                        &batch.adjacency_src,
                        &relation,
                        Some(&batch.adjacency_axis),
                        cells,
                        true,
                    )
                },
                || {
                    stable_edge_view(
                        "adjacency source",
                        &batch.adjacency_src,
                        &batch.adjacency_dst,
                        &relation,
                        Some(&batch.adjacency_axis),
                        cells,
                        false,
                    )
                },
            )
        },
        || {
            stable_edge_view(
                "adjacency relation",
                &relation,
                &batch.adjacency_src,
                &batch.adjacency_dst,
                Some(&batch.adjacency_axis),
                relation_count,
                true,
            )
        },
    );
    Ok(AdjacencyPlanArrays {
        relation,
        dst: dst?,
        src: src?,
        rel: rel?,
    })
}

fn radius_plans(
    batch: &PackedActBatch,
    cfg: &ActBuilderConfig,
) -> Result<RadiusPlanArrays, String> {
    if !cfg.use_occupied_radius_edges {
        return Ok(RadiusPlanArrays::default());
    }
    let rows = batch.radius_src.len();
    if batch.radius_dst.len() != rows
        || batch.radius_orbit.len() != rows
        || batch.radius_axis_or_neg1.len() != rows
    {
        return Err("packed radius columns have different row counts".to_owned());
    }
    let cells = batch.cell_occupancy.len();
    let relation_count = relation_vocabulary_size(cfg)?
        .checked_mul(2)
        .ok_or_else(|| "radius relation vocabulary size overflows usize".to_owned())?;
    if batch.radius_offsets.len() != batch.position_count + 1
        || batch.radius_offsets.first() != Some(&0)
        || batch.radius_offsets.last() != Some(&(rows as i64))
    {
        return Err("radius offsets disagree with the packed radius rows".to_owned());
    }
    let mut offsets = Vec::with_capacity(batch.radius_offsets.len());
    for position in 0..batch.position_count {
        let start = batch.radius_offsets[position];
        let end = batch.radius_offsets[position + 1];
        if end < start {
            return Err(format!("radius offsets decrease at position {position}"));
        }
        offsets
            .push(usize::try_from(start).map_err(|_| {
                format!("radius offset {start} at position {position} is negative")
            })?);
    }
    offsets.push(rows);
    let routed_counts: Vec<usize> = (0..batch.position_count)
        .into_par_iter()
        .map(|position| {
            batch.radius_axis_or_neg1[offsets[position]..offsets[position + 1]]
                .iter()
                .filter(|&&axis| axis >= 0)
                .count()
        })
        .collect();
    let routed_rows = routed_counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| "routed radius row count overflows usize".to_owned())
    })?;
    let mut routed_offsets = Vec::with_capacity(routed_counts.len() + 1);
    routed_offsets.push(0usize);
    for &count in &routed_counts {
        routed_offsets.push(
            routed_offsets
                .last()
                .expect("routed offsets have a leading zero")
                .checked_add(count)
                .ok_or_else(|| "routed radius row count overflows usize".to_owned())?,
        );
    }
    checked_i32("radius rows", rows)?;
    let mut relation = vec![0i64; rows];
    let mut axis_rows = vec![0i64; routed_rows];
    let mut routed_src = vec![0i64; routed_rows];
    let mut routed_dst = vec![0i64; routed_rows];
    let mut routed_relation = vec![0i64; routed_rows];
    let mut routed_axis = vec![0i64; routed_rows];
    let error = Mutex::new(None::<(usize, String)>);
    rayon::scope(|scope| {
        let mut relation_tail = relation.as_mut_slice();
        let mut row_tail = axis_rows.as_mut_slice();
        let mut source_tail = routed_src.as_mut_slice();
        let mut destination_tail = routed_dst.as_mut_slice();
        let mut routed_relation_tail = routed_relation.as_mut_slice();
        let mut axis_tail = routed_axis.as_mut_slice();
        for position in 0..batch.position_count {
            let start = offsets[position];
            let end = offsets[position + 1];
            let count = end - start;
            let routed_count = routed_counts[position];
            let (relation_rows, relation_rest) = relation_tail.split_at_mut(count);
            let (row_rows, row_rest) = row_tail.split_at_mut(routed_count);
            let (source_rows, source_rest) = source_tail.split_at_mut(routed_count);
            let (destination_rows, destination_rest) = destination_tail.split_at_mut(routed_count);
            let (routed_relation_rows, routed_relation_rest) =
                routed_relation_tail.split_at_mut(routed_count);
            let (axis_rows_out, axis_rest) = axis_tail.split_at_mut(routed_count);
            let error = &error;
            scope.spawn(move |_| {
                let mut routed = 0usize;
                for (local_row, relation_row) in relation_rows.iter_mut().enumerate() {
                    let row = start + local_row;
                    let failure = (|| {
                        let source =
                            checked_key("radius source", batch.radius_src[row], cells, row)?;
                        let colour = batch.cell_occupancy[source];
                        if !matches!(colour, 1 | 2) {
                            return Err(format!(
                                "radius source row {row} has occupancy {colour}, expected 1 or 2"
                            ));
                        }
                        let joint = batch.radius_orbit[row]
                            .checked_mul(2)
                            .and_then(|value| value.checked_add(i64::from(colour == 2)))
                            .ok_or_else(|| {
                                format!("radius relation at row {row} overflows i64")
                            })?;
                        checked_key("radius relation", joint, relation_count, row)?;
                        *relation_row = joint;
                        let axis = batch.radius_axis_or_neg1[row];
                        if !(-1..=2).contains(&axis) {
                            return Err(format!(
                                "radius row {row} has axis route {axis} outside -1..2"
                            ));
                        }
                        if axis >= 0 {
                            row_rows[routed] = i64::try_from(row).map_err(|_| {
                                format!("radius routed row {row} exceeds int64")
                            })?;
                            source_rows[routed] = batch.radius_src[row];
                            destination_rows[routed] = batch.radius_dst[row];
                            routed_relation_rows[routed] = joint;
                            axis_rows_out[routed] = axis;
                            routed += 1;
                        }
                        Ok::<(), String>(())
                    })();
                    if let Err(message) = failure {
                        let mut first = error.lock().expect("radius plan error mutex poisoned");
                        if first.as_ref().is_none_or(|current| row < current.0) {
                            *first = Some((row, message));
                        }
                        break;
                    }
                }
                if routed != routed_count {
                    let mut first = error.lock().expect("radius plan error mutex poisoned");
                    if first.as_ref().is_none_or(|current| start < current.0) {
                        *first = Some((
                            start,
                            format!(
                                "position {position} filled {routed} routed radius rows, expected {routed_count}"
                            ),
                        ));
                    }
                }
            });
            relation_tail = relation_rest;
            row_tail = row_rest;
            source_tail = source_rest;
            destination_tail = destination_rest;
            routed_relation_tail = routed_relation_rest;
            axis_tail = axis_rest;
        }
    });
    if let Some((_, message)) = error
        .into_inner()
        .expect("radius plan error mutex poisoned")
    {
        return Err(message);
    }
    let (((dst, src), rel), ((axis_dst, axis_src), axis_rel)) = join(
        || {
            let ((dst, src), rel) = join(
                || {
                    join(
                        || {
                            stable_edge_view(
                                "radius destination",
                                &batch.radius_dst,
                                &batch.radius_src,
                                &relation,
                                None,
                                cells,
                                true,
                            )
                        },
                        || {
                            stable_position_cell_view(
                                "radius source",
                                &offsets,
                                &batch.cell_offsets,
                                &batch.radius_src,
                                &batch.radius_dst,
                                &relation,
                                None,
                            )
                        },
                    )
                },
                || {
                    stable_position_relation_view(
                        "radius relation",
                        &offsets,
                        &relation,
                        &batch.radius_src,
                        &batch.radius_dst,
                        None,
                        relation_count,
                    )
                },
            );
            ((dst, src), rel)
        },
        || {
            let ((dst, src), rel) = join(
                || {
                    join(
                        || {
                            stable_edge_view(
                                "radius axis destination",
                                &routed_dst,
                                &routed_src,
                                &routed_relation,
                                Some(&routed_axis),
                                cells,
                                true,
                            )
                        },
                        || {
                            stable_position_cell_view(
                                "radius axis source",
                                &routed_offsets,
                                &batch.cell_offsets,
                                &routed_src,
                                &routed_dst,
                                &routed_relation,
                                Some(&routed_axis),
                            )
                        },
                    )
                },
                || {
                    stable_position_relation_view(
                        "radius axis relation",
                        &routed_offsets,
                        &routed_relation,
                        &routed_src,
                        &routed_dst,
                        Some(&routed_axis),
                        relation_count,
                    )
                },
            );
            ((dst, src), rel)
        },
    );
    Ok(RadiusPlanArrays {
        relation,
        dst: dst?,
        src: src?,
        rel: rel?,
        axis_rows,
        routed_src,
        routed_dst,
        routed_relation,
        routed_axis,
        axis_dst: axis_dst?,
        axis_src: axis_src?,
        axis_rel: axis_rel?,
    })
}

fn source_window_rows(values: &[i64], windows: usize) -> Result<SourceWindowPlanArrays, String> {
    checked_i32("action source-window", windows)?;
    checked_i32("action source-window", values.len())?;
    let mut counts = vec![0usize; windows];
    let mut live = Vec::with_capacity(values.len());
    let mut sentinel_rows = Vec::new();
    for (row, &value) in values.iter().enumerate() {
        if value == -1 {
            sentinel_rows.push(checked_i32("action sentinel rows", row)?);
        } else {
            let window = checked_key("action source-window", value, windows, row)?;
            counts[window] += 1;
            live.push((row, window));
        }
    }
    let ptr = csr_ptr("action source-window", &counts, live.len())?;
    let mut next: Vec<usize> = ptr[..windows].iter().map(|&value| value as usize).collect();
    let mut rows = vec![0i32; live.len()];
    for (row, window) in live {
        let output = next[window];
        next[window] += 1;
        rows[output] = checked_i32("action source-window", row)?;
    }
    Ok(SourceWindowPlanArrays {
        ptr,
        rows,
        sentinel_rows,
    })
}

fn gather_rows(values: &[i64], cells: usize) -> Result<GatherRowPlanArrays, String> {
    let source_count = cells
        .checked_add(1)
        .ok_or_else(|| "action base-cell source count overflows usize".to_owned())?;
    let sentinel = i64::try_from(cells)
        .map_err(|_| format!("cell count {cells} exceeds the int64 index frame"))?;
    let normalized: Vec<_> = values
        .iter()
        .map(|&value| if value >= 0 { value } else { sentinel })
        .collect();
    let plan = class_rows("action base cell", &normalized, source_count)?;
    Ok(GatherRowPlanArrays {
        ptr: plan.ptr,
        rows: plan.rows,
    })
}

fn row_positions(name: &str, offsets: &[i64]) -> Result<Vec<i64>, String> {
    if offsets.is_empty() || offsets[0] != 0 {
        return Err(format!("{name} offsets must start at zero"));
    }
    let total = usize::try_from(*offsets.last().expect("offsets are nonempty"))
        .map_err(|_| format!("{name} offsets end below zero"))?;
    let mut output = Vec::with_capacity(total);
    for position in 0..offsets.len() - 1 {
        let start = offsets[position];
        let end = offsets[position + 1];
        if end < start {
            return Err(format!("{name} offsets decrease at position {position}"));
        }
        let count =
            usize::try_from(end - start).map_err(|_| format!("{name} row count is negative"))?;
        let owner = i64::try_from(position)
            .map_err(|_| format!("{name} position {position} exceeds int64"))?;
        output.extend(std::iter::repeat_n(owner, count));
    }
    if output.len() != total {
        return Err(format!(
            "{name} owner rows total {}, expected {total}",
            output.len()
        ));
    }
    Ok(output)
}

fn phases(name: &str, owners: &[i64], phase_id: &[i64]) -> Result<Vec<i64>, String> {
    owners
        .iter()
        .enumerate()
        .map(|(row, &owner)| {
            let owner = checked_key(name, owner, phase_id.len(), row)?;
            Ok(phase_id[owner])
        })
        .collect()
}

fn latent_segments(
    name: &str,
    offsets: &[&[i64]],
    owners: &[&[i64]],
) -> Result<LatentSegmentArrays, String> {
    if offsets.is_empty() || offsets.len() != owners.len() {
        return Err(format!(
            "{name} needs matching nonempty offset and owner families"
        ));
    }
    let positions = offsets[0]
        .len()
        .checked_sub(1)
        .ok_or_else(|| format!("{name} offsets are empty"))?;
    if offsets.iter().any(|value| value.len() != positions + 1) {
        return Err(format!("{name} offset families disagree on position count"));
    }
    let mut bases = Vec::with_capacity(offsets.len());
    let mut base = 0i64;
    for family in offsets {
        bases.push(base);
        base = base
            .checked_add(*family.last().expect("offsets have positions"))
            .ok_or_else(|| format!("{name} family bases overflow i64"))?;
    }
    let mut ranges = Vec::with_capacity(positions * offsets.len() * 2);
    let mut range_base = Vec::with_capacity(positions * offsets.len());
    let mut counts = Vec::with_capacity(positions);
    for position in 0..positions {
        let mut within = 0usize;
        for (family, &family_offsets) in offsets.iter().enumerate() {
            let start = family_offsets[position];
            let end = family_offsets[position + 1];
            if end < start {
                return Err(format!(
                    "{name} family {family} offsets decrease at position {position}"
                ));
            }
            let global_start = bases[family]
                .checked_add(start)
                .ok_or_else(|| format!("{name} range start overflows i64"))?;
            let global_end = bases[family]
                .checked_add(end)
                .ok_or_else(|| format!("{name} range end overflows i64"))?;
            ranges.push(
                i32::try_from(global_start)
                    .map_err(|_| format!("{name} range start {global_start} exceeds int32"))?,
            );
            ranges.push(
                i32::try_from(global_end)
                    .map_err(|_| format!("{name} range end {global_end} exceeds int32"))?,
            );
            range_base.push(checked_i32(name, within)?);
            within = within
                .checked_add(
                    usize::try_from(end - start)
                        .map_err(|_| format!("{name} family {family} has negative row count"))?,
                )
                .ok_or_else(|| format!("{name} position row count overflows usize"))?;
        }
        counts.push(checked_i32(name, within)?);
    }
    let row_count: usize = owners.iter().map(|values| values.len()).sum();
    let mut row_pos = Vec::with_capacity(row_count);
    for family in owners {
        row_pos.extend_from_slice(family);
    }
    Ok(LatentSegmentArrays {
        ranges,
        range_base,
        counts,
        row_pos,
    })
}

fn embedding_plans(
    batch: &PackedActBatch,
) -> Result<
    (
        ClassRowPlanArrays,
        ClassRowPlanArrays,
        ClassRowPlanArrays,
        ClassRowPlanArrays,
        ClassRowPlanArrays,
    ),
    String,
> {
    let (((occupancy, legal), nearest), (pattern, status)) = join(
        || {
            let ((occupancy, legal), nearest) = join(
                || {
                    join(
                        || {
                            class_rows(
                                "cell occupancy",
                                &batch.cell_occupancy,
                                CELL_OCCUPANCY_CLASSES,
                            )
                        },
                        || class_rows("cell legal", &batch.cell_is_legal, CELL_LEGAL_CLASSES),
                    )
                },
                || {
                    class_rows(
                        "cell nearest",
                        &batch.cell_nearest_bucket,
                        CELL_NEAREST_CLASSES,
                    )
                },
            );
            ((occupancy, legal), nearest)
        },
        || {
            join(
                || {
                    class_rows(
                        "window pattern",
                        &batch.window_pattern_class,
                        WINDOW_PATTERN_CLASSES,
                    )
                },
                || class_rows("window status", &batch.window_status, WINDOW_STATUSES),
            )
        },
    );
    Ok((occupancy?, legal?, nearest?, pattern?, status?))
}

fn action_plans(
    batch: &PackedActBatch,
) -> Result<
    (
        ClassRowPlanArrays,
        ClassRowPlanArrays,
        SourceWindowPlanArrays,
        GatherRowPlanArrays,
    ),
    String,
> {
    let ((post1, pre_status), (source_window, base_cell)) = join(
        || {
            join(
                || class_rows("action post1", &batch.action_post1_class, POST1_CLASSES),
                || {
                    class_rows(
                        "action pre-status",
                        &batch.action_pre_status,
                        WINDOW_STATUSES,
                    )
                },
            )
        },
        || {
            join(
                || source_window_rows(&batch.action_window_index, batch.window_pattern_class.len()),
                || gather_rows(&batch.legal_to_cell_index, batch.cell_occupancy.len()),
            )
        },
    );
    Ok((post1?, pre_status?, source_window?, base_cell?))
}

struct OwnershipPlanArrays {
    cell_row_pos: Vec<i64>,
    window_row_pos: Vec<i64>,
    legal_row_pos: Vec<i64>,
    cell_phase: Vec<i64>,
    window_phase: Vec<i64>,
    action_phase: Vec<i64>,
    state_segment: LatentSegmentArrays,
    action_segment: LatentSegmentArrays,
}

fn ownership_plans(batch: &PackedActBatch) -> Result<OwnershipPlanArrays, String> {
    let ((cell_row_pos, window_row_pos), legal_row_pos) = join(
        || {
            join(
                || row_positions("cell", &batch.cell_offsets),
                || row_positions("window", &batch.window_offsets),
            )
        },
        || row_positions("legal", &batch.legal_offsets),
    );
    let cell_row_pos = cell_row_pos?;
    let window_row_pos = window_row_pos?;
    let legal_row_pos = legal_row_pos?;
    if batch.phase_id.len() != batch.position_count {
        return Err(format!(
            "phase_id has {} rows against {} positions",
            batch.phase_id.len(),
            batch.position_count
        ));
    }
    let ((cell_phase, window_phase), action_phase) = join(
        || {
            join(
                || phases("cell phase", &cell_row_pos, &batch.phase_id),
                || phases("window phase", &window_row_pos, &batch.phase_id),
            )
        },
        || phases("action phase", &legal_row_pos, &batch.phase_id),
    );
    let (state_segment, action_segment) = join(
        || {
            latent_segments(
                "state latent segments",
                &[&batch.cell_offsets, &batch.window_offsets],
                &[&cell_row_pos, &window_row_pos],
            )
        },
        || {
            latent_segments(
                "action latent segments",
                &[&batch.legal_offsets],
                &[&legal_row_pos],
            )
        },
    );
    Ok(OwnershipPlanArrays {
        cell_row_pos,
        window_row_pos,
        legal_row_pos,
        cell_phase: cell_phase?,
        window_phase: window_phase?,
        action_phase: action_phase?,
        state_segment: state_segment?,
        action_segment: action_segment?,
    })
}

/// Derive the complete, configuration-independent axis/routing plan superset.
///
/// Feature-disabled edge families use an all-empty closed wire prefix.  Axis
/// plans and radius routed subsets are otherwise built unconditionally; the
/// Python assembler selects the views its model configuration exposes.
pub fn build_plan_arrays(
    batch: &PackedActBatch,
    cfg: &ActBuilderConfig,
) -> Result<ActPlanArrays, String> {
    cfg.validate()?;
    let (((incidence, adjacency), radius), ((embedding, action), ownership)) = join(
        || {
            let ((incidence, adjacency), radius) = join(
                || join(|| incidence_plans(batch), || adjacency_plans(batch, cfg)),
                || radius_plans(batch, cfg),
            );
            ((incidence, adjacency), radius)
        },
        || {
            let (embedding, action) = join(|| embedding_plans(batch), || action_plans(batch));
            ((embedding, action), ownership_plans(batch))
        },
    );
    let (occupancy, legal, nearest, pattern, status) = embedding?;
    let (post1, pre_status, source_window, base_cell) = action?;
    let ownership = ownership?;
    Ok(ActPlanArrays {
        incidence: incidence?,
        adjacency: adjacency?,
        radius: radius?,
        class_cell_occupancy: occupancy,
        class_cell_legal: legal,
        class_cell_nearest: nearest,
        class_window_pattern: pattern,
        class_window_status: status,
        class_action_post1: post1,
        class_action_pre_status: pre_status,
        action_source_window: source_window,
        action_base_cell: base_cell,
        cell_row_pos: ownership.cell_row_pos,
        window_row_pos: ownership.window_row_pos,
        legal_row_pos: ownership.legal_row_pos,
        cell_phase: ownership.cell_phase,
        window_phase: ownership.window_phase,
        action_phase: ownership.action_phase,
        state_segment: ownership.state_segment,
        action_segment: ownership.action_segment,
    })
}

/// Build, collate, and plan engine positions without crossing into Python.
pub fn build_planned_batch(
    positions: &[hexo_engine::Position],
    cfg: &ActBuilderConfig,
) -> Result<PlannedActBatch, String> {
    let packed = act_encoder::build_packed_batch(positions, cfg)?;
    let plans = build_plan_arrays(&packed, cfg)?;
    Ok(PlannedActBatch { packed, plans })
}

/// Replay prefixes, build, collate, and plan them without crossing into Python.
pub fn build_planned_batch_prefixes(
    games: &[Vec<(i16, i16)>],
    ts: &[usize],
    cfg: &ActBuilderConfig,
) -> Result<PlannedActBatch, String> {
    let packed = act_encoder::build_packed_batch_prefixes(games, ts, cfg)?;
    let plans = build_plan_arrays(&packed, cfg)?;
    Ok(PlannedActBatch { packed, plans })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::act_encoder::{CellScope, WindowScope};

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

    fn stable_order(keys: &[i64]) -> Vec<usize> {
        let mut rows: Vec<_> = (0..keys.len()).collect();
        rows.sort_by_key(|&row| keys[row]);
        rows
    }

    #[test]
    fn stable_edge_csr_preserves_input_order_within_ties() {
        let key = [2, 0, 2, 1, 0, 2];
        let left = [10, 11, 12, 13, 14, 15];
        let right = [20, 21, 22, 23, 24, 25];
        let axis = [0, 1, 2, 0, 1, 2];
        let plan = stable_edge_view("test", &key, &left, &right, Some(&axis), 4, false).unwrap();
        assert_eq!(plan.ptr, [0, 2, 3, 6, 6]);
        assert_eq!(plan.left, [11, 14, 13, 10, 12, 15]);
        assert_eq!(plan.right, [21, 24, 23, 20, 22, 25]);
        assert_eq!(plan.axis, [1, 1, 0, 0, 2, 2]);
    }

    #[test]
    fn sorted_edge_view_refuses_a_decrease() {
        let error =
            stable_edge_view("test", &[0, 2, 1], &[0; 3], &[0; 3], None, 3, true).unwrap_err();
        assert!(error.contains("declares sorted keys"), "{error}");
    }

    #[test]
    fn position_parallel_views_equal_global_stable_scatter_and_refuse_crossing() {
        let row_offsets = [0usize, 6, 10];
        let cell_offsets = [0i64, 4, 7];
        let cell_key = [2, 0, 2, 1, 3, 0, 5, 4, 6, 4];
        let relation_key = [2, 0, 2, 1, 0, 2, 1, 2, 0, 1];
        let left = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        let right = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        let axis = [0, 1, 2, 0, 1, 2, 2, 1, 0, 2];

        assert_eq!(
            stable_position_cell_view(
                "test cell",
                &row_offsets,
                &cell_offsets,
                &cell_key,
                &left,
                &right,
                Some(&axis),
            )
            .unwrap(),
            stable_edge_view("test cell", &cell_key, &left, &right, Some(&axis), 7, false,)
                .unwrap(),
        );
        assert_eq!(
            stable_position_relation_view(
                "test relation",
                &row_offsets,
                &relation_key,
                &left,
                &right,
                Some(&axis),
                3,
            )
            .unwrap(),
            stable_edge_view(
                "test relation",
                &relation_key,
                &left,
                &right,
                Some(&axis),
                3,
                false,
            )
            .unwrap(),
        );

        let mut crossing_key = cell_key;
        crossing_key[6] = 2;
        let error = stable_position_cell_view(
            "test crossing",
            &row_offsets,
            &cell_offsets,
            &crossing_key,
            &left,
            &right,
            None,
        )
        .unwrap_err();
        assert!(error.contains("crosses packed position"), "{error}");

        let error = stable_position_cell_view(
            "test offsets",
            &row_offsets,
            &[0, 8, 7],
            &cell_key,
            &left,
            &right,
            None,
        )
        .unwrap_err();
        assert!(error.contains("cell offsets are invalid"), "{error}");
    }

    #[test]
    fn class_rows_are_stable_and_blocks_never_cross_a_class() {
        let mut values = Vec::new();
        values.extend(std::iter::repeat_n(2, 130));
        values.push(0);
        values.extend(std::iter::repeat_n(2, 127));
        values.push(0);
        let plan = class_rows("test", &values, 4).unwrap();
        assert_eq!(plan.ptr, [0, 2, 2, 259, 259]);
        assert_eq!(plan.rows[..2], [130, 258]);
        assert_eq!(plan.rows[2], 0);
        assert_eq!(plan.rows[258], 257);
        assert_eq!(plan.block_ptr, [0, 1, 1, 4, 4]);
        assert_eq!(plan.block_starts, [0, 2, 130, 258]);
        assert_eq!(plan.block_lengths, [2, 128, 128, 1]);
    }

    #[test]
    fn source_window_rows_partition_live_and_sentinel_rows_stably() {
        let plan = source_window_rows(&[-1, 2, 0, 2, -1, 1], 4).unwrap();
        assert_eq!(plan.ptr, [0, 1, 2, 4, 4]);
        assert_eq!(plan.rows, [2, 5, 1, 3]);
        assert_eq!(plan.sentinel_rows, [0, 4]);
    }

    #[test]
    fn latent_segments_match_the_python_multi_range_layout() {
        let cell_offsets = [0, 2, 3];
        let window_offsets = [0, 1, 4];
        let cell_owner = [0, 0, 1];
        let window_owner = [0, 1, 1, 1];
        let plan = latent_segments(
            "test",
            &[&cell_offsets, &window_offsets],
            &[&cell_owner, &window_owner],
        )
        .unwrap();
        assert_eq!(plan.ranges, [0, 2, 3, 4, 2, 3, 4, 7]);
        assert_eq!(plan.range_base, [0, 2, 0, 1]);
        assert_eq!(plan.counts, [3, 4]);
        assert_eq!(plan.row_pos, [0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn complete_planned_batch_has_exact_stable_views_and_closed_disabled_prefixes() {
        let games = vec![vec![], vec![(0, 0)], vec![(0, 0), (1, 0), (2, 0), (0, 1)]];
        let ts = vec![0, 1, 4];
        let cfg = full_config();
        let planned = build_planned_batch_prefixes(&games, &ts, &cfg).unwrap();
        let batch = &planned.packed;
        let plans = &planned.plans;
        let cells = batch.cell_occupancy.len();
        let windows = batch.window_pattern_class.len();
        let radius_rows = batch.radius_src.len();
        let incidence_rows = batch
            .window_incidence_mask
            .iter()
            .filter(|&&value| value)
            .count();

        assert_eq!(plans.incidence.cell.len(), incidence_rows);
        assert_eq!(plans.incidence.dst.ptr.len(), windows + 1);
        assert_eq!(plans.incidence.src.ptr.len(), cells + 1);
        assert_eq!(plans.incidence.rel.ptr.len(), INCIDENCE_RELATIONS + 1);
        assert_eq!(
            plans.incidence.dst.left,
            plans
                .incidence
                .cell
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            plans.incidence.dst.right,
            plans
                .incidence
                .relation
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            plans.incidence.dst.axis,
            plans
                .incidence
                .axis
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
        );
        let cell_order = stable_order(&plans.incidence.cell);
        assert_eq!(
            plans.incidence.src.left,
            cell_order
                .iter()
                .map(|&row| plans.incidence.window[row] as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            plans.incidence.src.right,
            cell_order
                .iter()
                .map(|&row| plans.incidence.relation[row] as i32)
                .collect::<Vec<_>>()
        );

        assert_eq!(plans.radius.relation.len(), radius_rows);
        assert_eq!(plans.radius.dst.ptr.len(), cells + 1);
        assert_eq!(plans.radius.dst.left.len(), radius_rows);
        assert_eq!(plans.radius.axis_rows.len(), plans.radius.routed_src.len());
        assert_eq!(
            plans.radius.axis_rows,
            batch
                .radius_axis_or_neg1
                .iter()
                .enumerate()
                .filter_map(|(row, &axis)| (axis >= 0).then_some(row as i64))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            plans.class_action_post1.rows.len(),
            batch.action_post1_class.len()
        );
        assert_eq!(plans.cell_row_pos.len(), cells);
        assert_eq!(plans.window_row_pos.len(), windows);
        assert_eq!(plans.state_segment.row_pos.len(), cells + windows);
        assert_eq!(plans.state_segment.ranges.len(), batch.position_count * 4);

        let mut disabled = cfg;
        disabled.use_cell_adjacency = false;
        disabled.use_occupied_radius_edges = false;
        let disabled = build_planned_batch_prefixes(&games, &ts, &disabled).unwrap();
        assert_eq!(disabled.plans.adjacency, AdjacencyPlanArrays::default());
        assert_eq!(disabled.plans.radius, RadiusPlanArrays::default());
    }
}
