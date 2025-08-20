const DEBRUIJN64: u64 = 0x03f79d71b4cb0a89;
const RANK_8: u64 = 0xFF00_0000_0000_0000;
const RANK_1: u64 = 0x0000_0000_0000_00FF;
const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const INDEX64: [u32; 64] = [
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
fn trailing_zeros_debruijn(x: u64) -> u64 {
    debug_assert!(x != 0 && x.count_ones() == 1);
    let idx = (x.wrapping_mul(DEBRUIJN64)) >> 58;
    INDEX64[idx]
}


#[inline(always)]
fn increment_rank(source: u64) -> u64 {
    (!RANK_8 & source) << 8
}


#[inline(always)]
fn decrement_rank(source: u64) -> u64 {
    (!RANK_1 & source) >> 8
}


#[inline(always)]
fn move_right(source: u64) -> u64 {
    (source & !FILE_H) << 1
}


#[inline(always)]
fn move_left(source: u64) -> u64 {
    (source & !FILE_A) >> 1
}


#[repr(usize)]  // guarantees discriminant values start from 0
pub enum ChesspieceType {
    WhiteRooks,
    WhiteKnights,
    WhiteBishops,
    WhiteQueens,
    WhiteKing,
    WhitePawns,
    BlackRooks,
    BlackKnights,
    BlackBishops,
    BlackQueens,
    BlackKing,
    BlackPawns,
}

impl ChesspieceType {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub const ALL: [ChesspieceType; 12] = [
        ChesspieceType::WhiteRooks,
        ChesspieceType::WhiteKnights,
        ChesspieceType::WhiteBishops,
        ChesspieceType::WhiteQueens,
        ChesspieceType::WhiteKing,
        ChesspieceType::WhitePawns,
        ChesspieceType::BlackRooks,
        ChesspieceType::BlackKnights,
        ChesspieceType::BlackBishops,
        ChesspieceType::BlackQueens,
        ChesspieceType::BlackKing,
        ChesspieceType::BlackPawns,
    ];
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
    pub fn get(&self, piece: ChesspieceType) -> u64 {
        self.pieces[piece.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, piece: ChesspieceType) -> &mut u64 {
        &mut self.pieces[piece.index()]
    }
}
