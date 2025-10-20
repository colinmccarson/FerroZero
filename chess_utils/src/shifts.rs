use crate::consts::*;
use crate::utils::branchless_select;


#[inline(always)]
pub const fn increment_rank(source: u64, num_ranks: u64) -> u64 {
    branchless_select(num_ranks == 0, (!RANK_8 & source) << (num_ranks << 3), source)
}


#[inline(always)]
pub const fn decrement_rank(source: u64, num_ranks: u64) -> u64 {
    branchless_select(num_ranks == 0, (!RANK_1 & source) >> (num_ranks << 3), source)
}


#[inline(always)]
pub const fn move_right(source: u64, num_files: u64) -> u64 {
    let row = 0xFFu64 << (((source.trailing_zeros() & 0x3F) >> 3) << 3);
    branchless_select(num_files == 0, (source >> num_files) & row, source)
}


#[inline(always)]
pub const fn move_left(source: u64, num_files: u64) -> u64 {
    let row = 0xFFu64 << (((source.trailing_zeros() & 0x3F) >> 3) << 3);
    branchless_select(num_files == 0, (source << num_files) & row, source)
}


#[inline(always)]
pub const fn move_source(source: u64, vertical: i32, horizontal: i32) -> u64 {
    let horizontal_moves = [move_left(source, horizontal.abs() as u64), move_right(source, horizontal.abs() as u64)];
    let horizontal_result = horizontal_moves[(horizontal >= 0) as usize];
    let vertical_moves = [decrement_rank(horizontal_result, vertical.abs() as u64), increment_rank(horizontal_result, vertical.abs() as u64)];
    vertical_moves[(vertical >= 0) as usize]
}


#[cfg(test)]
mod tests {
    use crate::shifts::{decrement_rank, increment_rank, move_left, move_right};
    use crate::utils::{get_file_index, get_rank_index};

    #[test]
    fn test_increment_rank() {
        for i in 0u64..64 {
            let source: u64 = 1u64 << i;
            let source_rank = get_rank_index(source).unwrap();
            let source_file = get_file_index(source).unwrap();
            for amount_increment in 1u64..8 {
                let move_u64: u64 = increment_rank(1u64 << i, amount_increment);
                if source_rank + amount_increment > 7 {
                    assert_eq!(move_u64, 0u64);
                } else {
                    let actual_move_rank = get_rank_index(move_u64).unwrap();
                    let actual_move_file = get_file_index(move_u64).unwrap();
                    assert_eq!(actual_move_file, source_file);
                    assert_eq!(actual_move_rank, source_rank + amount_increment);
                }
            }
        }
    }

    #[test]
    fn test_decrement_rank() {
        for i in 0u64..64 {
            let source: u64 = 1u64 << i;
            let source_rank = get_rank_index(source).unwrap();
            let source_file = get_file_index(source).unwrap();
            for amount_decrement in 1u64..8 {
                let move_u64: u64 = decrement_rank(1u64 << i, amount_decrement);
                if amount_decrement > source_rank {
                    assert_eq!(move_u64, 0u64);
                } else {
                    let actual_move_rank = get_rank_index(move_u64).unwrap();
                    let actual_move_file = get_file_index(move_u64).unwrap();
                    assert_eq!(actual_move_file, source_file);
                    assert_eq!(actual_move_rank, source_rank - amount_decrement);
                }
            }
        }
    }

    #[test]
    fn test_move_left() {
        for i in 0u64..64 {
            let source: u64 = 1u64 << i;
            let source_rank = get_rank_index(source).unwrap();
            let source_file = get_file_index(source).unwrap();
            for amount_left in 0u64..8 {
                let move_u64: u64 = move_left(source, amount_left);
                if amount_left > source_file {
                    assert_eq!(move_u64, 0u64);
                } else {
                    let actual_move_rank = get_rank_index(move_u64).unwrap();
                    let actual_move_file = get_file_index(move_u64).unwrap();
                    assert_eq!(actual_move_rank, source_rank);
                    assert_eq!(actual_move_file, source_file - amount_left);
                }
            }
        }
    }

    #[test]
    fn test_move_right() {
        for i in 0u64..64 {
            let source: u64 = 1u64 << i;
            let source_rank = get_rank_index(source).unwrap();
            let source_file = get_file_index(source).unwrap();
            for amount_right in 0u64..8 {
                let move_u64: u64 = move_right(source, amount_right);
                if source_file + amount_right > 7 {
                    assert_eq!(move_u64, 0u64);
                } else {
                    let actual_move_rank = get_rank_index(move_u64).unwrap();
                    let actual_move_file = get_file_index(move_u64).unwrap();
                    assert_eq!(actual_move_rank, source_rank);
                    assert_eq!(actual_move_file, source_file + amount_right);
                }
            }
        }
    } 
}
