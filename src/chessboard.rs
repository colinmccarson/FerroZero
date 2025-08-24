use crate::chess_attacks as chess_attacks;

const DEBRUIJN64: u64 = 0x03f79d71b4cb0a89;
const RANK_8: u64 = 0xFF00_0000_0000_0000;
const RANK_1: u64 = 0x0000_0000_0000_00FF;
const INDEX64: [u64; 64] = [
    0,  1, 48,  2, 57, 49, 28,  3,
    61, 58, 50, 42, 38, 29, 17,  4,
    62, 55, 59, 36, 53, 51, 43, 22,
    45, 39, 33, 30, 24, 18, 12,  5,
    63, 47, 56, 27, 60, 41, 37, 16,
    54, 35, 52, 21, 44, 32, 23, 11,
    46, 26, 40, 15, 34, 20, 31, 10,
    25, 14, 19,  9, 13,  8,  7,  6,
];


/** The following return zero when there is an error **/
#[inline(always)]
pub const fn trailing_zeros_debruijn(x: u64) -> u64 {
    debug_assert!(x != 0);
    debug_assert!(x.count_ones() == 1);
    let idx: usize = ((x.wrapping_mul(DEBRUIJN64)) >> 58) as usize;
    INDEX64[idx]
}


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


#[repr(usize)]  // guarantees discriminant values start from 0
pub enum ChesspieceOffset {
    Rooks,
    Knights,
    Bishops,
    Queens,
    King,
    Pawns,
}


impl ChesspieceOffset {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [ChesspieceOffset; 6] = [
        ChesspieceOffset::Rooks,
        ChesspieceOffset::Knights,
        ChesspieceOffset::Bishops,
        ChesspieceOffset::Queens,
        ChesspieceOffset::King,
        ChesspieceOffset::Pawns,
    ];
}


pub enum ChessColor {
    White = 0,
    Black = 6,
}


struct Chessboard {
    pieces: [u64; 12],
}


impl Chessboard {
    /**
    Board convention is: rank = floor(log8(loc)), file = log2(loc - floor(loc8(loc)))
    **/
    pub fn new() -> Chessboard {
        Chessboard {
            pieces: [0; 12],
        }
    }

    #[inline]
    pub fn get(&self, piece: ChesspieceOffset) -> u64 {
        self.pieces[piece.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, piece: ChesspieceOffset) -> &mut u64 {
        &mut self.pieces[piece.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_increment_rank() {
        assert_eq!(increment_rank(2, 1), 0x00000200);
        assert_eq!(decrement_rank(2, 3), 0);
    }
}
