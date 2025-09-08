// build.rs
use std::{env, fs, path::PathBuf};


macro_rules! make_pext_table {
    ($attack_fn:path, $n:expr) => {
        {
            let mut tbl = [[0u64; $n]; 64];
            let mut sq_ind = 0usize;
            while sq_ind < 64 {
                let mut i = 0usize;
                while i < $n {
                    tbl[sq_ind][i] = $attack_fn(1u64 << sq_ind, i as u64);
                    i += 1;
                }
                sq_ind += 1;
            }
            tbl
        }
    };
}


macro_rules! make_chessboard_constant_from_fill {
    ($fill_fn:path) => {
        {
            let mut tbl = [0u64; 64];
            let mut sq_ind = 0usize;
            while sq_ind < 64 {
                tbl[sq_ind] = $fill_fn(1u64 << sq_ind);
                sq_ind += 1;
            }
            tbl
        }
    }
}


//noinspection ALL
fn main() {
    // Out directory where Cargo expects build artifacts
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let rook_moves = rook_moves();
    let knight_moves = knight_moves();
    let bishop_moves = bishop_moves();
    let queen_moves = queen_moves();
    let king_moves = king_moves();


    let rook_pext_table = make_pext_table!(rook_attack_single, 1 << 12);
    let bishop_pext_table = make_pext_table!(bishop_attack_single, 1 << 11);
    let rook_moves_without_ray_ends = make_chessboard_constant_from_fill!(rook_no_ray_ends);
    let chess_tables_ready = "chess tables ready";

    let code = format!(
        "const HELLO: &str = {chess_tables_ready:?};\n\
         pub static ROOK_PEXT_TABLE: [[u64; 4096]; 64] = {rook_pext_table:?};\n\
         pub static BISHOP_PEXT_TABLE: [[u64; 2048]; 64] = {bishop_pext_table:?};\n\
         pub static ROOK_MOVES_NO_RAY_ENDS: [u64; 64] = {rook_moves_without_ray_ends:?};\n\
         pub static ROOK_MOVES: [u64; 64] = {rook_moves:?};\n\
         pub static KNIGHT_MOVES: [u64; 64] = {knight_moves:?};\n\
         pub static BISHOP_MOVES: [u64; 64] = {bishop_moves:?};\n\
         pub static QUEEN_MOVES: [u64; 64] = {queen_moves:?};\n\
         pub static KING_MOVES: [u64; 64] = {king_moves:?};\n",
    );

    fs::write(out_dir.join("generated_chess_tables.rs"), code).unwrap();
}


use core::arch::x86_64::_pext_u64;

use chess_utils::utils::*;
use chess_utils::shifts::*;
use chess_utils::consts::*;

// ---------- Single-square generators (const) ----------

fn rook_single(start: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut j: u64 = 1;
    while j <= 7 {
        acc |= increment_rank(start, j);
        acc |= decrement_rank(start, j);
        acc |= move_left(start, j);
        acc |= move_right(start, j);
        j += 1;
    }
    acc
}


fn rook_no_ray_ends(start: u64) -> u64 {
    let r8_tile: u64 = start << (8 * (7u64.saturating_sub(get_rank_index(start).unwrap())));
    let r1_tile: u64 = start >> (8 * get_rank_index(start).unwrap());
    let fa_tile: u64 = start << get_file_index(start).unwrap();
    let fh_tile: u64 = start >> (7u64.saturating_sub(get_file_index(start).unwrap()));

    rook_single(start) & !(r8_tile | r1_tile | fa_tile | fh_tile)
}


fn bishop_single(start: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut j: u64 = 1;
    while j <= 7 {
        acc |= increment_rank(move_left(start, j), j); // up-left
        acc |= increment_rank(move_right(start, j), j); // up-right
        acc |= decrement_rank(move_left(start, j), j); // down-left
        acc |= decrement_rank(move_right(start, j), j); // down-right
        j += 1;
    }
    acc
}


 fn knight_single(sq: u64) -> u64 {
    let nw1 = increment_rank(move_left(sq, 2), 1);
    let nw2 = increment_rank(move_left(sq, 1), 2);
    let ne2 = increment_rank(move_right(sq, 1), 2);
    let ne1 = increment_rank(move_right(sq, 2), 1);
    let se1 = decrement_rank(move_right(sq, 2), 1);
    let se2 = decrement_rank(move_right(sq, 1), 2);
    let sw2 = decrement_rank(move_left(sq, 1), 2);
    let sw1 = decrement_rank(move_left(sq, 2), 1);

    nw1 | nw2 | ne1 | ne2 | se1 | se2 | sw1 | sw2
}


 fn queen_single(sq: u64) -> u64 {
    rook_single(sq) | bishop_single(sq)
}


 fn king_single(sq: u64) -> u64 {
    let left = move_left(sq, 1);
    let up = increment_rank(sq, 1);
    let right = move_right(sq, 1);
    let down = decrement_rank(sq, 1);
    let upleft = move_left(up, 1);
    let upright = move_right(up, 1);
    let downleft = move_left(down, 1);
    let downright = move_right(down, 1);

    left | upleft | up | upright | right | downright | down | downleft
}


macro_rules! make_table {
    ($name:ident, $gen:ident) => {
        pub fn $name() -> [u64; 64] {
            let mut arr = [0u64; 64];
            let mut i = 0usize;
            while i < 64 {
                arr[i] = $gen(1u64 << (i as u32));
                i += 1;
            }
            arr
        }
    };
}


make_table!(rook_moves, rook_single);
make_table!(bishop_moves, bishop_single);
make_table!(queen_moves, queen_single);
make_table!(knight_moves, knight_single);
make_table!(king_moves, king_single);


fn rebuild_occupancy_mask(mask: u64, pext_result: u64) -> u64 {
    let mut tmp = mask;
    let mut occ = 0u64;
    for idx in 0usize..(mask.count_ones() as usize) {
        let occ_sq = 1u64 << tmp.trailing_zeros();
        if (pext_result >> idx) & 1 != 0 {
            occ |= occ_sq;
        }
        tmp &= tmp - 1;
    }
    occ
}


fn rook_attack_single(sq: u64, pext_result: u64) -> u64 {
    // we exclude the outer rim
    let mask = rook_no_ray_ends(sq);  // never more than 12 for a rook; no need to consider ends in pext
    // exclude the edges; at runtime we check occ & edges
    if pext_result == 0 {
        return rook_single(sq);
    }
   
    let occ = rebuild_occupancy_mask(mask, pext_result);
    brute_force_rook_attack(sq, occ)
}


fn bishop_attack_single(sq: u64, pext_result: u64) -> u64 {
    let mask = bishop_single(sq) & !BOARD_EDGES;  // never more than 11 for a bishop
    if pext_result == 0 {
        return bishop_single(sq);
    }

    let occ = rebuild_occupancy_mask(mask, pext_result);
    brute_force_bishop_attack(sq, occ)
}
