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


#[inline(always)]
fn trailing_zeros_debruijn(x: u64) -> u64 {
    debug_assert!(x != 0 && x.count_ones() == 1);
    let idx = ((x.wrapping_mul(DEBRUIJN64)) >> 58);
    INDEX64[idx]
}


#[inline(always)]
fn increment_rank_saturating(source: u64) -> u64 {
    ((!RANK_8 & source) << 8) | (source & RANK_8)
}


#[inline(always)]
fn decrement_rank_saturating(source: u64) -> u64 {
    ((!RANK_1 & source) >> 8) | (source & RANK_1)
}


#[inline(always)]
fn move_right(source: u64) -> u64 {
    (source & !FILE_H) << 1 | (source & FILE_H)
}


#[inline(always)]
fn move_left(source: u64) -> u64 {
    (source & !FILE_A) >> 1 | (source & FILE_A)
}


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


#[derive(Clone)]
pub struct Chessboard {
    white_rooks: u64,
    white_knights: u64,
    white_bishops: u64,
    white_queens: u64,
    white_king: u64,
    white_pawns: u64,
    black_rooks: u64,
    black_knights: u64,
    black_bishops: u64,
    black_queens: u64,
    black_king: u64,
    black_pawns: u64,
}


impl Chessboard {
    /**
    Board convention is: rank = floor(log8(loc)), file = log2(loc - floor(loc8(loc)))
    **/
    pub fn new() -> Chessboard {
        Chessboard {
            white_rooks: 0,
            white_knights: 0,
            white_bishops: 0,
            white_queens: 0,
            white_king: 0,
            white_pawns: 0,
            black_rooks: 0,
            black_knights: 0,
            black_bishops: 0,
            black_queens: 0,
            black_king: 0,
            black_pawns: 0,
        }
    }

    pub fn get(&self, bb: ChesspieceType) -> u64 {
        match bb {
            ChesspieceType::WhiteRooks => self.white_rooks,
            ChesspieceType::WhiteKnights => self.white_knights,
            ChesspieceType::WhiteBishops => self.white_bishops,
            ChesspieceType::WhiteQueens => self.white_queens,
            ChesspieceType::WhiteKing => self.white_king,
            ChesspieceType::WhitePawns => self.white_pawns,
            ChesspieceType::BlackRooks => self.black_rooks,
            ChesspieceType::BlackKnights => self.black_knights,
            ChesspieceType::BlackBishops => self.black_bishops,
            ChesspieceType::BlackQueens => self.black_queens,
            ChesspieceType::BlackKing => self.black_king,
            ChesspieceType::BlackPawns => self.black_pawns,
        }
    }

    pub fn get_mut(&mut self, bb: ChesspieceType) -> &mut u64 {
        match bb {
            ChesspieceType::WhiteRooks => &mut self.white_rooks,
            ChesspieceType::WhiteKnights => &mut self.white_knights,
            ChesspieceType::WhiteBishops => &mut self.white_bishops,
            ChesspieceType::WhiteQueens => &mut self.white_queens,
            ChesspieceType::WhiteKing => &mut self.white_king,
            ChesspieceType::WhitePawns => &mut self.white_pawns,
            ChesspieceType::BlackRooks => &mut self.black_rooks,
            ChesspieceType::BlackKnights => &mut self.black_knights,
            ChesspieceType::BlackBishops => &mut self.black_bishops,
            ChesspieceType::BlackQueens => &mut self.black_queens,
            ChesspieceType::BlackKing => &mut self.black_king,
            ChesspieceType::BlackPawns => &mut self.black_pawns,
        }
    }

}
