#[allow(long_running_const_eval)]
use unroll::unroll_for_loops;

use crate::consts;
use crate::shifts;


pub fn print_u64_as_8x8_bit_string(n: u64) {
    let binary_string = format!("{:064b}", n);

    for (i, c) in binary_string.chars().enumerate() {
        print!("{}", c);
        if (i + 1) % 8 == 0 {
            print!("\n");
        }
        if (i + 1) % 64 == 0 {
            println!();
        }
    }
}


pub const fn get_rank_index(sq: u64) -> Option<u64> {
    let mut i = 0u64;
    while i < 8 {
        if sq & (0xFF << (i << 3)) > 0 {
            return Some(i);
        }
        i += 1;
    }
    return None;
}


pub const fn get_file_index(mv: u64) -> Option<u64> {
    let mut i = 0u64;
    while i < 64 {
        if mv & (1u64 << i) > 0 {
            return Some(7 - (i % 8));
        }
        i += 1;
    }
    return None;
}


pub fn collect_coordinates(mv_set: u64) -> Vec<(u64, u64)> {
    let mut v: Vec<(u64, u64)> = Vec::new();
    for i in 0u64..64 {
        let mv: u64 = 1u64 << i;
        if (mv & mv_set) > 0 {
            v.push((get_rank_index(mv).unwrap(), get_file_index(mv).unwrap()));
        }
    }
    v
}


pub fn get_occ_indices_from_bitboard(mut board: u64) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    while board > 0 {
        let ind = board.trailing_zeros() as usize;
        let sq = 1u64 << ind;
        board &= !sq;
        out.push(ind);
    }
    out
}


pub fn map_rank_and_file_to_sq(rank: u64, file: u64) -> u64 {
    if rank > 7 || file > 7 {
        return 0;
    }
    1u64 << (8 * rank + (7 - file))
}


pub fn get_rank_and_file_diff(sq: u64, target: u64) -> (i64, i64) {

    let sq_rank = get_rank_index(sq).unwrap() as i64;
    let sq_file = get_file_index(sq).unwrap() as i64;

    let occ_rank = get_rank_index(target).unwrap() as i64;
    let occ_file = get_file_index(target).unwrap() as i64;

    let rank_diff = occ_rank - sq_rank;
    let file_diff = occ_file - sq_file;
    (rank_diff, file_diff)
}


pub fn get_k_behind_straight_line_if_any(sq: u64, occ: u64, k: i64) -> u64 {
    let sq_rank = get_rank_index(sq).unwrap() as i64;
    let sq_file = get_file_index(sq).unwrap() as i64;
    let (mut rank_diff, mut file_diff) = get_rank_and_file_diff(sq, occ);
    
    if rank_diff != 0 && file_diff != 0 {
        return 0;
    }
    
    if rank_diff > 0 {
        rank_diff += k;
    }
    else if rank_diff < 0 {
        rank_diff -= k;
    }
    else if file_diff > 0 {
        file_diff += k;
    }
    else {
        file_diff -= k;
    }
    map_rank_and_file_to_sq(std::cmp::max(sq_rank + rank_diff, 0) as u64, std::cmp::max(sq_file + file_diff, 0) as u64)
}


//#[unroll_for_loops]
#[macro_export]
macro_rules! zip {
    ($a:expr, $b:expr, $n:expr) => {{
        let a = &$a;
        let b = &$b;
        let mut out = [(a[0], b[0]); $n];
        for i in 1..$n {
            out[i] = (a[i], b[i]);
        }
        out
    }};
}


pub fn map_direction_to_tuple(dir: &crate::consts::DIRECTIONS) -> (i32, i32) {
    match dir {
        crate::consts::DIRECTIONS::N  => ( 1,  0),
        crate::consts::DIRECTIONS::NE => ( 1,  1),
        crate::consts::DIRECTIONS::E  => ( 0,  1),
        crate::consts::DIRECTIONS::SE => (-1,  1),
        crate::consts::DIRECTIONS::S  => (-1,  0),
        crate::consts::DIRECTIONS::SW => (-1, -1),
        crate::consts::DIRECTIONS::W  => ( 0, -1),
        crate::consts::DIRECTIONS::NW => ( 1, -1),
    }
}



pub fn get_direction_to_target(sq: u64, target: u64) -> Option<crate::consts::DIRECTIONS> {
    let sq_rank = get_rank_index(sq).unwrap() as i64;
    let sq_file = get_file_index(sq).unwrap() as i64;

    let target_rank = get_rank_index(target).unwrap() as i64;
    let target_file = get_file_index(target).unwrap() as i64;

    let dr = target_rank - sq_rank;
    let df = target_file - sq_file;

    match (dr.signum(), df.signum()) {
        (1, 0)  => Some(crate::consts::DIRECTIONS::N),
        (1, 1)  => Some(crate::consts::DIRECTIONS::NE),
        (0, 1)  => Some(crate::consts::DIRECTIONS::E),
        (-1, 1) => Some(crate::consts::DIRECTIONS::SE),
        (-1, 0) => Some(crate::consts::DIRECTIONS::S),
        (-1, -1)=> Some(crate::consts::DIRECTIONS::SW),
        (0, -1) => Some(crate::consts::DIRECTIONS::W),
        (1, -1) => Some(crate::consts::DIRECTIONS::NW),
        _ => None,
    }
}



pub fn brute_force_rook_attack(sq: u64, occ: u64) -> u64 {
    let ray_n = attack_in_dir(sq, occ, 1, 0);
    let ray_s = attack_in_dir(sq, occ, -1, 0);
    let ray_e = attack_in_dir(sq, occ, 0, 1);
    let ray_w = attack_in_dir(sq, occ, 0, -1);
    
    (ray_n | ray_s | ray_e | ray_w) & !sq
}


fn in_bounds(rank: i32, file: i32) -> bool {
    0 <= rank && rank < 8 && 0 <= file && file < 8
}


/// Attacks in a direction from sq up to and including the first blocker in the direction given by occ
pub fn attack_in_dir(sq: u64, occ: u64, vertical: i32, horizontal: i32) -> u64 {
    let mut rank = get_rank_index(sq).unwrap() as i32;
    let mut file = get_file_index(sq).unwrap() as i32;
    let mut k = 1i32;
    let mut attack = 0u64;
    while in_bounds(rank + k * vertical, file + k * horizontal) && (attack & occ == 0) {
        attack |= shifts::move_source(sq, k * vertical, k * horizontal);
        k += 1;
    }
    attack
}


pub fn attack_in_dir_enum(sq: u64, occ: u64, dir: &crate::consts::DIRECTIONS) -> u64 {
    let (v, h) = map_direction_to_tuple(dir);
    attack_in_dir(sq, occ, v, h)
}


pub fn brute_force_bishop_attack(sq: u64, occ: u64) -> u64 {
    let ray_nw = attack_in_dir(sq, occ, 1, -1);
    let ray_ne = attack_in_dir(sq, occ, 1, 1);
    let ray_se = attack_in_dir(sq, occ, -1, 1);
    let ray_sw = attack_in_dir(sq, occ, -1, -1);

    ray_nw | ray_ne | ray_se | ray_sw
}


pub fn shift_safe_files(mut board: u64, k: i32) -> u64 {
    for i in 0..k.abs() {
        if k > 0 {
            board = (board >> 1) & !consts::FILE_A;  // prevent rollover
        }
        else {
            board = (board << 1) & !consts::FILE_H;
        }
    }
    board
}
