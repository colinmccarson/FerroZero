use crate::chessboard;


const ROOK_MOVES: [u64; 64] = rook_moves();

const KNIGHT_MOVES: [u64; 64] = knight_moves();

const BISHOP_MOVES: [u64; 64] = bishop_moves();

const QUEEN_MOVES: [u64; 64] = queen_moves();

const KING_MOVES: [u64; 64] = king_moves();

const WHITE_PAWN_MOVES: [u64; 64] = white_pawn_moves();

const BLACK_PAWN_MOVES: [u64; 64] = black_pawn_moves();

// ---------- Single-square generators (const) ----------

pub const fn rook_single(start: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut j: u64 = 1;
    while j <= 7 {
        acc |= chessboard::increment_rank(start, j);
        acc |= chessboard::decrement_rank(start, j);
        acc |= chessboard::move_left(start, j);
        acc |= chessboard::move_right(start, j);
        j += 1;
    }
    acc
}

const fn bishop_single(start: u64) -> u64 {
    let mut acc: u64 = 0;
    let mut j: u64 = 1;
    while j <= 7 {
        acc |= chessboard::increment_rank(chessboard::move_left(start, j), j); // up-left
        acc |= chessboard::increment_rank(chessboard::move_right(start, j), j); // up-right
        acc |= chessboard::decrement_rank(chessboard::move_left(start, j), j); // down-left
        acc |= chessboard::decrement_rank(chessboard::move_right(start, j), j); // down-right
        j += 1;
    }
    acc
}

const fn knight_single(sq: u64) -> u64 {
    let nw1 = chessboard::increment_rank(chessboard::move_left(sq, 2), 1);
    let nw2 = chessboard::increment_rank(chessboard::move_left(sq, 1), 2);
    let ne2 = chessboard::increment_rank(chessboard::move_right(sq, 1), 2);
    let ne1 = chessboard::increment_rank(chessboard::move_right(sq, 2), 1);
    let se1 = chessboard::decrement_rank(chessboard::move_right(sq, 2), 1);
    let se2 = chessboard::decrement_rank(chessboard::move_right(sq, 1), 2);
    let sw2 = chessboard::decrement_rank(chessboard::move_left(sq, 1), 2);
    let sw1 = chessboard::decrement_rank(chessboard::move_left(sq, 2), 1);

    nw1 | nw2 | ne1 | ne2 | se1 | se2 | sw1 | sw2
}

const fn queen_single(sq: u64) -> u64 {
    rook_single(sq) | bishop_single(sq)
}

const fn king_single(sq: u64) -> u64 {
    let left = chessboard::move_left(sq, 1);
    let up = chessboard::increment_rank(sq, 1);
    let right = chessboard::move_right(sq, 1);
    let down = chessboard::decrement_rank(sq, 1);
    let upleft = chessboard::move_left(up, 1);
    let upright = chessboard::move_right(up, 1);
    let downleft = chessboard::move_left(down, 1);
    let downright = chessboard::move_right(down, 1);

    left | upleft | up | upright | right | downright | down | downleft
}

const fn white_pawn_single(sq: u64) -> u64 {
    let up = chessboard::increment_rank(sq, 1);
    let upleft = chessboard::move_left(up, 1);
    let upright = chessboard::move_right(up, 1);
    let usual = up | upleft | upright;
    if sq & 0xFF00u64 > 0 {  // first move
        return usual | chessboard::increment_rank(up, 1);
    }
    usual
}

const fn black_pawn_single(sq: u64) -> u64 {
    let down = chessboard::decrement_rank(sq, 1);
    let downleft = chessboard::move_left(down, 1);
    let downright = chessboard::move_right(down, 1);
    let usual = down | downleft | downright;
    if sq & 0x00FF_0000_0000_0000u64 > 0 { // first move
        return usual | chessboard::decrement_rank(down, 1);
    }
    usual
}


macro_rules! make_table {
    ($name:ident, $gen:ident) => {
        pub const fn $name() -> [u64; 64] {
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


make_table!(rook_moves,   rook_single);
make_table!(bishop_moves, bishop_single);
make_table!(queen_moves,  queen_single);
make_table!(knight_moves, knight_single);
make_table!(king_moves,   king_single);
make_table!(white_pawn_moves, white_pawn_single);
make_table!(black_pawn_moves, black_pawn_single);


#[cfg(test)]
mod tests {
    use crate::util::*;
    use super::*;

    #[test]
    fn test_rook_moves() {
        for (i, mv_set) in ROOK_MOVES.clone().into_iter().enumerate() {
            let rank0 = get_rank_index(1u64 << i).unwrap();
            let file0 = get_file_index(1u64 << i).unwrap();
            for (rank, file) in collect_coordinates(mv_set) {
                assert!((rank == rank0) | (file == file0));
            }
            assert_eq!(mv_set.count_ones(), 14);
        }
    }

    #[test]
    fn test_bishop_moves() {
        for (i, mv_set) in BISHOP_MOVES.clone().into_iter().enumerate() {
            // y - y0 = x - x0 or y - y0 = x0 - x
            let x0 = get_file_index(1u64 << i).unwrap() as i64;
            let y0 = get_rank_index(1u64 << i).unwrap() as i64;
            for (y, x) in collect_coordinates(mv_set) {
                let (x, y): (i64, i64) = (x as i64, y as i64);
                assert!((y - y0 == x - x0) || (y - y0 == x0 - x));
            }
            assert!(7 <= mv_set.count_ones() && mv_set.count_ones() <= 14);
        }
    }

    #[test]
    fn test_queen_moves() {
        // correctness of rook & bishop moves ~should~ imply correctness of queen moves
        for (i, mv_set) in QUEEN_MOVES.clone().into_iter().enumerate() {
            assert_eq!(mv_set, ROOK_MOVES[i] | BISHOP_MOVES[i]);
        }
    }

    #[test]
    fn test_king_moves() {
        for (i, mv_set) in KING_MOVES.clone().into_iter().enumerate() {
            let x0 = get_file_index(1u64 << i).unwrap();
            let y0 = get_rank_index(1u64 << i).unwrap();
            let mv_vec_coords = collect_coordinates(mv_set);

            let xm = x0.saturating_sub(1);
            let ym = y0.saturating_sub(1);
            let xp = std::cmp::min(x0 + 1, 7);
            let yp = std::cmp::min(y0 + 1, 7);

            let king_moves: Vec<(u64, u64)> = vec![
                (ym, xm),
                (ym, x0),
                (ym, xp),
                (y0, xm),
                (y0, xp),
                (yp, xm),
                (yp, x0),
                (yp, xp),
            ].into_iter().filter(|(y, x): &(u64, u64)| { (*y, *x) != (y0, x0) }).collect();

            let correct_moves_set: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::from_iter(king_moves);
            let mv_set_coords: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::from_iter(mv_vec_coords);
            assert_eq!(mv_set_coords, correct_moves_set);
        }
    }

    #[test]
    fn test_white_pawn_moves() {
        for (i, mv_set) in WHITE_PAWN_MOVES.clone().into_iter().enumerate() {
            let pawn_rank = get_rank_index(1u64 << i).unwrap();
            let pawn_file = get_file_index(1u64 << i).unwrap();
            let actual_moves: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::from_iter(
                collect_coordinates(mv_set)
            );
            if pawn_rank == 1 {
                let expected_moves: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::from_iter(
                    vec![(2, pawn_file), (2, pawn_file.saturating_sub(1)), (2, std::cmp::min(pawn_file + 1, 7)), (3, pawn_file)]
                );
                assert_eq!(expected_moves, actual_moves);
            }
            else if pawn_rank < 7 {
                let expected_moves: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::from_iter(
                    vec![(pawn_rank + 1, pawn_file.saturating_sub(1)), (pawn_rank + 1, pawn_file), (pawn_rank + 1, std::cmp::min(pawn_file + 1, 7))]
                );
                assert_eq!(expected_moves, actual_moves);
            }
            else {
                assert_eq!(mv_set, 0u64);
            }
        }
    }

    #[test]
    fn test_black_pawn_moves() {
        for (i, mv_set) in BLACK_PAWN_MOVES.clone().into_iter().enumerate() {
            let pawn_rank = get_rank_index(1u64 << i).unwrap();
            let pawn_file = get_file_index(1u64 << i).unwrap();
            let actual_moves: std::collections::HashSet<(u64, u64)> =
                std::collections::HashSet::from_iter(collect_coordinates(mv_set));
            if pawn_rank == 6 {
                let expected_moves: std::collections::HashSet<(u64, u64)> =
                    std::collections::HashSet::from_iter(vec![
                        (5, pawn_file.saturating_sub(1)),
                        (5, pawn_file),
                        (5, std::cmp::min(pawn_file + 1, 7)),
                        (4, pawn_file),
                    ]);
                assert_eq!(expected_moves, actual_moves);
            } else if pawn_rank > 0 {
                let expected_moves: std::collections::HashSet<(u64, u64)> =
                    std::collections::HashSet::from_iter(vec![
                        (pawn_rank - 1, pawn_file.saturating_sub(1)),
                        (pawn_rank - 1, pawn_file),
                        (pawn_rank - 1, std::cmp::min(pawn_file + 1, 7)),
                    ]);
                assert_eq!(expected_moves, actual_moves);
            } else {
                assert_eq!(mv_set, 0u64);
            }
        }
    }
}
