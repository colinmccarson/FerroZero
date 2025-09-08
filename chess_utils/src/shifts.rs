use crate::consts::*;


#[inline(always)]
pub const fn increment_rank(source: u64, num_ranks: u64) -> u64 {
    (!RANK_8 & source) << (num_ranks << 3)
}


#[inline(always)]
pub const fn decrement_rank(source: u64, num_ranks: u64) -> u64 {
    (!RANK_1 & source) >> (num_ranks << 3)
}


#[inline(always)]
pub const fn move_right(source: u64, num_files: u64) -> u64 {
    let row = 0xFFu64 << (((source.trailing_zeros() & 0x3F) >> 3) << 3);
    (source >> num_files) & row
}


#[inline(always)]
pub const fn move_left(source: u64, num_files: u64) -> u64 {
    let row = 0xFFu64 << (((source.trailing_zeros() & 0x3F) >> 3) << 3);
    (source << num_files) & row
}


#[inline(always)]
pub const fn move_source(source: u64, vertical: i32, horizontal: i32) -> u64 {
    let horizontal_moves = [move_left(source, horizontal.abs() as u64), move_right(source, horizontal.abs() as u64)];
    let horizontal_result = horizontal_moves[(horizontal >= 0) as usize];
    let vertical_moves = [decrement_rank(horizontal_result, vertical.abs() as u64), increment_rank(horizontal_result, vertical.abs() as u64)];
    vertical_moves[(vertical >= 0) as usize]
}
