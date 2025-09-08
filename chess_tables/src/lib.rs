use std::arch::x86_64::_pext_u64;
use gen_tables::*;


pub fn queen_attack_single(sq: u64, occ: u64) -> u64 {
    let sq_ind = sq.trailing_zeros() as usize;
    let mask_bishop = BISHOP_MOVES[sq_ind];
    let pext_result_bishop = unsafe { _pext_u64(occ, mask_bishop) } as usize;

    let mask_rook = ROOK_MOVES[sq_ind];
    let pext_result_rook = unsafe { _pext_u64(occ, mask_rook) } as usize;

    BISHOP_PEXT_TABLE[sq_ind][pext_result_bishop] | ROOK_PEXT_TABLE[sq_ind][pext_result_rook]
}



#[cfg(test)]
mod tests {
    use crate::*;
    use core::arch::x86_64::_pext_u64;
    use rand::Rng;
    use rand::prelude::*;
    use chess_utils::utils::*;
    use chess_utils::consts::*;
    use chess_utils::shifts::*;
    
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
    fn test_knight_moves() {
        for (i, mv_set) in KNIGHT_MOVES.clone().into_iter().enumerate() {
            let mv: u64 = 1u64 << i;
            let rank0 = get_rank_index(mv).unwrap() as i64;
            let file0 = get_file_index(mv).unwrap() as i64;
            assert!(2 <= mv_set.count_ones() && mv_set.count_ones() <= 8);
            for (rank, file) in collect_coordinates(mv_set) {
                let rank = rank as i64;
                let file = file as i64;
                assert!(
                    (((rank - rank0).abs() == 2) && ((file - file0).abs() == 1))
                        || ((rank - rank0).abs() == 1) && ((file - file0).abs() == 2)
                );
            }
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
            ]
                .into_iter()
                .filter(|(y, x): &(u64, u64)| (*y, *x) != (y0, x0))
                .collect();

            let correct_moves_set: std::collections::HashSet<(u64, u64)> =
                std::collections::HashSet::from_iter(king_moves);
            let mv_set_coords: std::collections::HashSet<(u64, u64)> =
                std::collections::HashSet::from_iter(mv_vec_coords);
            assert_eq!(mv_set_coords, correct_moves_set);
        }
    }
    
    #[test]
    fn test_rook_no_ray_ends() {
        for i in 0usize..64 {
            let sq = 1u64 << i;
            let no_ray_ends = ROOK_MOVES_NO_RAY_ENDS[i];
            let ones_diff: u32 = no_ray_ends.count_ones().abs_diff(ROOK_MOVES[i].count_ones());
            if sq & BOARD_CORNERS > 0 {
                assert_eq!(ones_diff, 2);
            }
            else if sq & BOARD_EDGES > 0 {
                assert_eq!(ones_diff, 3);
            }
            else {
                assert_eq!(ones_diff, 4);
            }
            assert_eq!(no_ray_ends & ROOK_MOVES[i], no_ray_ends);
        }
    }

    #[test]
    fn test_rook_pext() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for (i, &mask) in ROOK_MOVES_NO_RAY_ENDS.iter().enumerate() {
            let sq = 1u64 << i;
            for _ in 0..20 {
                let occ_bits: u64 = rng.random::<u64>() & mask;
                let needs_pext = ((mask & occ_bits) != 0) as usize; // TODO wrong; mask needs to be not including the ends of rays
                if needs_pext == 0 {
                    continue;
                }
                let pext_result = unsafe { _pext_u64(occ_bits, mask) } as usize;
                let resulting_rays = ROOK_PEXT_TABLE[i][pext_result];  // typically a cross-like shape from sq
                let expected = brute_force_rook_attack(sq, occ_bits);
                if resulting_rays != expected {
                    print_u64_as_8x8_bit_string(sq);
                    print_u64_as_8x8_bit_string(mask);
                    print_u64_as_8x8_bit_string(pext_result as u64);
                    print_u64_as_8x8_bit_string(occ_bits);
                    print_u64_as_8x8_bit_string(expected);
                    print_u64_as_8x8_bit_string(resulting_rays);
                }
                assert_eq!(resulting_rays, expected);
            }
        }
    }

    #[test]
    fn test_bishop_pext() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for (i, &mask) in BISHOP_MOVES.iter().enumerate() {
            let mask = mask & !BOARD_EDGES;
            let sq = 1u64 << i;
            for _ in 0..20 {
                let occ_bits: u64 = rng.random::<u64>() & mask;
                let needs_pext = ((mask & occ_bits) != 0) as usize; // TODO wrong; mask needs to be not including the ends of rays
                if needs_pext == 0 {
                    continue;
                }
                let pext_result = unsafe { _pext_u64(occ_bits, mask) } as usize;
                let resulting_rays = BISHOP_PEXT_TABLE[i][pext_result];
                let expected = brute_force_bishop_attack(sq, occ_bits);
                if resulting_rays != expected {
                    print_u64_as_8x8_bit_string(sq);
                    print_u64_as_8x8_bit_string(mask);
                    print_u64_as_8x8_bit_string(pext_result as u64);
                    print_u64_as_8x8_bit_string(occ_bits);
                    print_u64_as_8x8_bit_string(expected);
                    print_u64_as_8x8_bit_string(resulting_rays);
                }
                assert_eq!(resulting_rays, expected);
            }
        }
    }

    // #[test]
    // fn test_white_pawn_moves() {
    //     for (i, mv_set) in WHITE_PAWN_MOVES.clone().into_iter().enumerate() {
    //         let pawn_rank = get_rank_index(1u64 << i).unwrap();
    //         let pawn_file = get_file_index(1u64 << i).unwrap();
    //         let actual_moves: std::collections::HashSet<(u64, u64)> =
    //             std::collections::HashSet::from_iter(collect_coordinates(mv_set));
    //         if pawn_rank == 1 {
    //             let expected_moves: std::collections::HashSet<(u64, u64)> =
    //                 std::collections::HashSet::from_iter(vec![
    //                     (2, pawn_file),
    //                     (2, pawn_file.saturating_sub(1)),
    //                     (2, std::cmp::min(pawn_file + 1, 7)),
    //                     (3, pawn_file),
    //                 ]);
    //             assert_eq!(expected_moves, actual_moves);
    //         } else if pawn_rank < 7 {
    //             let expected_moves: std::collections::HashSet<(u64, u64)> =
    //                 std::collections::HashSet::from_iter(vec![
    //                     (pawn_rank + 1, pawn_file.saturating_sub(1)),
    //                     (pawn_rank + 1, pawn_file),
    //                     (pawn_rank + 1, std::cmp::min(pawn_file + 1, 7)),
    //                 ]);
    //             assert_eq!(expected_moves, actual_moves);
    //         } else {
    //             assert_eq!(mv_set, 0u64);
    //         }
    //     }
    // }
    //
    // #[test]
    // fn test_black_pawn_moves() {
    //     for (i, mv_set) in BLACK_PAWN_MOVES.clone().into_iter().enumerate() {
    //         let pawn_rank = get_rank_index(1u64 << i).unwrap();
    //         let pawn_file = get_file_index(1u64 << i).unwrap();
    //         let actual_moves: std::collections::HashSet<(u64, u64)> =
    //             std::collections::HashSet::from_iter(collect_coordinates(mv_set));
    //         if pawn_rank == 6 {
    //             let expected_moves: std::collections::HashSet<(u64, u64)> =
    //                 std::collections::HashSet::from_iter(vec![
    //                     (5, pawn_file.saturating_sub(1)),
    //                     (5, pawn_file),
    //                     (5, std::cmp::min(pawn_file + 1, 7)),
    //                     (4, pawn_file),
    //                 ]);
    //             assert_eq!(expected_moves, actual_moves);
    //         } else if pawn_rank > 0 {
    //             let expected_moves: std::collections::HashSet<(u64, u64)> =
    //                 std::collections::HashSet::from_iter(vec![
    //                     (pawn_rank - 1, pawn_file.saturating_sub(1)),
    //                     (pawn_rank - 1, pawn_file),
    //                     (pawn_rank - 1, std::cmp::min(pawn_file + 1, 7)),
    //                 ]);
    //             assert_eq!(expected_moves, actual_moves);
    //         } else {
    //             assert_eq!(mv_set, 0u64);
    //         }
    //     }
    // }
}
