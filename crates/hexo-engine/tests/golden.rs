//! Frozen golden vectors for the two things that are pinned across processes: the
//! Zobrist hash and the canonical legal-move ordering.

mod common;

use common::{check_all_oracles, zobrist_oracle};
use hexo_engine::{ACTION_ORDER_VERSION, Action, HexCoord, Position, RULES_VERSION};

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
}

/// A fixed 40-ply game that spreads across 36 rows and 68 columns — several arena
/// growths — and does not terminate.
const GAME: [(i16, i16); 40] = [
    (0, 0),
    (5, 2),
    (9, 4),
    (11, -3),
    (-3, -4),
    (1, -2),
    (7, 9),
    (-10, 2),
    (6, -7),
    (19, -8),
    (11, 13),
    (4, -3),
    (21, -10),
    (13, 8),
    (21, -14),
    (-7, -8),
    (10, 20),
    (21, -8),
    (0, -10),
    (17, 18),
    (11, -1),
    (2, 27),
    (-6, 34),
    (-8, -15),
    (1, 27),
    (16, 13),
    (7, -11),
    (-6, 39),
    (-15, -14),
    (11, 24),
    (-4, -21),
    (20, -9),
    (16, 22),
    (7, -3),
    (14, -13),
    (13, 26),
    (8, 8),
    (-11, 47),
    (4, 2),
    (-5, -7),
];

/// `Position::zobrist()` after each ply of [`GAME`].
const ZOBRIST_BY_PLY: [u64; 40] = [
    0xfbfe_1397_1f36_aa01,
    0x1f1d_59c8_7475_865d,
    0x2b8b_88e6_bbe4_bb83,
    0x7e19_48a1_9433_041d,
    0x60b8_f35e_b732_8b4b,
    0x226e_81f3_41ae_04c4,
    0x5152_5cdc_245e_c6df,
    0x3474_b30b_19a1_1527,
    0x972a_24ae_bf09_a795,
    0x690e_fc2d_83b8_eaa8,
    0xe2d5_e398_71aa_5bee,
    0x9220_0fee_e0ce_737f,
    0x12b9_0a24_d295_03f3,
    0x2789_14b4_dae2_16dc,
    0x5b54_42d8_bc27_adb6,
    0xce20_4c9c_72df_2f1c,
    0xf5f7_4f54_5987_11bb,
    0x2591_fd65_7193_701c,
    0x6f04_a570_ff74_9473,
    0x5737_f057_d0d4_0b95,
    0x98b8_6923_7736_531f,
    0x4560_1fe6_cc86_95eb,
    0x1978_f4cd_e6dd_3978,
    0x5c5a_e33a_4b33_4f8b,
    0x5cf0_6693_214d_4bb3,
    0x3759_6087_37a1_c649,
    0xb977_a5fc_1516_53c1,
    0x7936_8246_1c2e_d38b,
    0x8673_bd23_ff98_7c32,
    0xb6b5_7199_ae58_b4c8,
    0x4550_c4aa_a54a_c809,
    0xc3b4_a881_bc3e_0a6d,
    0xdc9f_6357_cd39_b2c8,
    0x51bc_e92d_e762_72b4,
    0x9830_948e_6eff_929c,
    0x6eb0_c294_75f6_7f2b,
    0x9c4c_04c9_237d_9a06,
    0x8ebc_8327_577a_d7f3,
    0xadb9_04d8_1475_8a83,
    0x0075_7703_90be_692f,
];

/// A 64-bit digest of the full legal-move ordering emitted at each ply.
const ORDER_DIGEST_BY_PLY: [u64; 40] = [
    0x6cf1_f1fe_a5ac_1e7e,
    0x9cdd_6cb3_053f_e111,
    0xe523_5b00_216f_627d,
    0x5e1c_64ad_f9b4_d284,
    0xc75a_f6ae_0342_cad2,
    0xf566_e612_a02c_e6e2,
    0x85b5_6c3d_a23d_a45f,
    0x1b74_17e9_8d43_5495,
    0x880b_2505_fc51_7333,
    0x5dcb_54b2_16d9_0844,
    0xb915_d20e_4294_2bd1,
    0xbb35_67a2_fda8_bc38,
    0x34da_596d_958b_0c60,
    0x5166_5448_3920_dfff,
    0x93bb_d5ef_13c7_a50d,
    0x7259_a3b7_2d63_5961,
    0x717d_db7d_47db_3485,
    0x1809_7bae_4e4c_3c8d,
    0x33ce_a6e4_c483_ca80,
    0x3783_5d8e_f5da_0a0b,
    0x81f2_0ba8_cb43_d77b,
    0x4848_48d5_b377_a364,
    0xc67e_01bb_4ccd_82d2,
    0xf332_b869_9219_d223,
    0xfdca_d8ff_2241_3a45,
    0xc5ae_a12f_5472_57d9,
    0x50ad_e442_7cd5_f36e,
    0x882d_0dff_2815_0c5c,
    0x75ce_124f_a997_864a,
    0x91bf_4497_2eea_5dbf,
    0x288c_6c1f_e12c_9e62,
    0x0486_c373_3757_1ba4,
    0x4681_7c97_79df_ac6b,
    0xbf34_ff0b_b2f9_1b13,
    0xf094_7642_2afd_47a1,
    0x86f7_1ddc_7937_7b1f,
    0x12f2_38dd_7515_dc72,
    0xec91_b94a_7a1f_79fd,
    0x50ec_f9c0_5adb_eb70,
    0xbfb4_b985_a930_5457,
];

/// Order-sensitive digest of one ply's legal-move list.
fn order_digest(pos: &Position) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in pos.legal_actions() {
        h ^= u64::from(a.id().0);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// Replay [`GAME`] and collect both per-ply streams.
fn streams() -> (Vec<u64>, Vec<u64>) {
    let mut pos = Position::new();
    let mut zs = Vec::new();
    let mut ds = Vec::new();
    for &(q, r) in &GAME {
        pos.advance(act(q, r))
            .expect("the golden game must be legal");
        zs.push(pos.zobrist());
        ds.push(order_digest(&pos));
    }
    (zs, ds)
}

#[test]
fn the_golden_game_is_legal_and_does_not_terminate() {
    let mut pos = Position::new();
    for (ply, &(q, r)) in GAME.iter().enumerate() {
        pos.advance(act(q, r))
            .unwrap_or_else(|e| panic!("golden ply {ply} ({q}, {r}) rejected: {e}"));
        assert!(!pos.is_terminal(), "golden game terminated at ply {ply}");
        check_all_oracles(&pos, ply + 1);
    }
    assert_eq!(pos.stone_count(), GAME.len() as u32);
}

#[test]
fn zobrist_per_ply_matches_the_frozen_table() {
    let (zs, _) = streams();
    assert_eq!(
        zs.as_slice(),
        ZOBRIST_BY_PLY.as_slice(),
        "RULES_VERSION {RULES_VERSION}: the Zobrist stream moved. \
         Regenerate deliberately and bump the version.\nactual: {zs:#018x?}"
    );
}

#[test]
fn legal_move_ordering_per_ply_matches_the_frozen_table() {
    let (_, ds) = streams();
    assert_eq!(
        ds.as_slice(),
        ORDER_DIGEST_BY_PLY.as_slice(),
        "ACTION_ORDER_VERSION {ACTION_ORDER_VERSION}: the canonical ordering moved. \
         Regenerate deliberately and bump the version.\nactual: {ds:#018x?}"
    );
}

#[test]
fn zobrist_stream_matches_the_independent_oracle() {
    let mut pos = Position::new();
    for (ply, &(q, r)) in GAME.iter().enumerate() {
        pos.advance(act(q, r)).expect("legal");
        assert_eq!(
            pos.zobrist(),
            zobrist_oracle(&pos),
            "oracle disagreement at ply {ply}"
        );
        assert_eq!(pos.zobrist(), ZOBRIST_BY_PLY[ply], "table at ply {ply}");
    }
}

/// [`Position::legal_rank`] of each played move of [`GAME`], at the ply it was played.
const PLAYED_RANK_BY_PLY: [usize; GAME.len()] = [
    0, 184, 279, 330, 56, 220, 375, 16, 477, 809, 660, 429, 973, 764, 989, 148, 771, 1236, 398,
    1150, 884, 505, 262, 251, 806, 1496, 1083, 453, 41, 1501, 668, 1996, 1845, 1373, 1715, 1705,
    1440, 355, 1346, 756,
];

#[test]
fn played_move_ranks_match_the_frozen_table() {
    let mut pos = Position::new();
    let mut ranks = Vec::new();
    for &(q, r) in &GAME {
        ranks.push(pos.legal_rank(act(q, r)).expect("the played move is legal"));
        pos.advance(act(q, r)).expect("legal");
    }
    assert_eq!(
        ranks.as_slice(),
        PLAYED_RANK_BY_PLY.as_slice(),
        "ACTION_ORDER_VERSION {ACTION_ORDER_VERSION}: the action index moved. \
         Every existing checkpoint's policy head is now wrong. Regenerate \
         deliberately and bump the version.\nactual: {ranks:?}"
    );
}

#[test]
fn nth_legal_inverts_the_frozen_ranks() {
    let mut pos = Position::new();
    for (ply, &(q, r)) in GAME.iter().enumerate() {
        let rank = PLAYED_RANK_BY_PLY[ply];
        assert_eq!(
            pos.nth_legal(rank),
            Some(act(q, r)),
            "ply {ply}: nth_legal({rank}) is not the played move"
        );
        pos.advance(act(q, r)).expect("legal");
    }
}

#[test]
fn the_canonical_ordering_is_ascending_action_ids_at_every_ply() {
    let mut pos = Position::new();
    for (ply, &(q, r)) in GAME.iter().enumerate() {
        pos.advance(act(q, r)).expect("legal");
        let ids: Vec<u32> = pos.legal_actions().map(|a| a.id().0).collect();
        assert_eq!(ids.len(), pos.legal_count(), "ply {ply}");
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "ply {ply}: ordering is not strictly ascending");
        }
    }
}
