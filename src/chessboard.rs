use core::arch::x86_64::_pext_u64;

use chess_tables;
use chess_utils::consts::*;
use chess_utils::shifts::*;
use chess_utils::utils::*;
use gen_tables::*;


#[repr(usize)]
#[derive(Clone, Copy)]
pub enum Colors {
    WHITE = 0usize,
    BLACK = 1usize,
}

const DEBRUIJN64: u64 = 0x03f79d71b4cb0a89;
const INDEX64: [u64; 64] = [
    0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45,
    39, 33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46,
    26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
];

/** The following return zero when there is an error **/
#[inline(always)]
pub const fn trailing_zeros_debruijn(x: u64) -> u64 {
    debug_assert!(x != 0);
    debug_assert!(x.count_ones() == 1);
    let idx: usize = ((x.wrapping_mul(DEBRUIJN64)) >> 58) as usize;
    INDEX64[idx]
}


#[derive(Clone, Copy)]
struct Move {
    source: u64,
    dests: u64,
    color: Colors,
    enpassant_location: u64,
}


impl Move {
    #[inline]
    pub fn new(source: u64, dests: u64, color: Colors, enpassant_location: u64) -> Move {
        Move { source, dests, color, enpassant_location }
    }
    
    #[inline]
    pub fn is_legal(&self) -> bool {
        self.source != 0
    }
}


#[inline]
/// Take a source, which it will remove the next piece from (in lsb -> msb order)
/// And returns a tuple (index: usize, piece_sq: u64)
/// If source is zero, returns 0 for piece_sq, signaling that there are no further pieces.
fn get_next_piece_as_u64_and_rm_from_source(source: &mut u64) -> (usize, u64) {
    let piece_ind = source.trailing_zeros() as usize;
    let is_not_zero = (piece_ind != 64) as u64;
    let piece_ind = piece_ind * is_not_zero as usize;
    let piece_sq = (1u64 << (piece_ind as u64 * is_not_zero)) * is_not_zero;
    *source &= !piece_sq;
    (piece_ind, piece_sq)
}


struct Chessboard {
    white_pawns: u64,
    white_rooks: u64,
    white_knights: u64,
    white_bishops: u64,
    white_queens: u64,
    white_king: u64,
    black_pawns: u64,
    black_rooks: u64,
    black_knights: u64,
    black_bishops: u64,
    black_queens: u64,
    black_king: u64,
    enpassant_location: u64  // bitboard square where enpassant would happen
}

impl Chessboard {
    /**
    Board convention is: rank = floor(log8(loc)), file = log2(loc - floor(loc8(loc)))
    **/
    pub fn new() -> Chessboard {
        let white_pawns = 0xFF00u64;
        let white_rooks = 0x81u64;
        let white_knights = 0x42u64;
        let white_bishops = 0x24u64;
        let white_queens = 0x8u64;
        let white_king = 0x4u64;

        let black_pawns = white_pawns << (8 * 5);
        let black_rooks = white_rooks << (8 * 7);
        let black_knights = white_knights << (8 * 7);
        let black_bishops = white_bishops << (8 * 7);
        let black_queens = white_queens << (8 * 7);
        let black_king = white_king << (8 * 7);

        Chessboard {
            white_pawns: white_pawns,
            white_rooks: white_rooks,
            white_knights: white_knights,
            white_bishops: white_bishops,
            white_queens: white_queens,
            white_king: white_king,
            black_pawns: black_pawns,
            black_rooks: black_rooks,
            black_knights: black_knights,
            black_bishops: black_bishops,
            black_queens: black_queens,
            black_king: black_king,
            enpassant_location: 0,
        }
    }
    
    pub fn new_blank() -> Chessboard {
        Chessboard {
            white_pawns: 0,
            white_rooks: 0,
            white_knights: 0,
            white_bishops: 0,
            white_queens: 0,
            white_king: 0,
            black_pawns: 0,
            black_rooks: 0,
            black_knights: 0,
            black_bishops: 0,
            black_queens: 0,
            black_king: 0,
            enpassant_location: 0,
        }
    }

    #[inline]
    fn get_combined_white_pieces(&self) -> u64 {
        self.white_rooks | self.white_knights | self.white_bishops | self.white_queens | self.white_king | self.white_pawns
    }

    #[inline]
    fn get_combined_black_pieces(&self) -> u64 {
        self.black_rooks | self.black_knights | self.black_bishops | self.black_queens | self.black_king | self.black_pawns
    }
    
    fn generate_pseudolegal_white_pawn_moves(&self) -> [Move; 8] {
        let black_occ = self.get_combined_black_pieces();
        let all_occ = self.get_combined_white_pieces() | black_occ;
        let mut source = self.white_pawns;
        let mut all_possible_moves = [Move::new(0, 0, Colors::WHITE, 0); 8];
        for i in 0..8 {
            let (pawn_ind, pawn_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = pawn_sq;
            all_possible_moves[i].dests = (pawn_sq >> 8) | ((pawn_sq & RANK_2) >> 16);
            all_possible_moves[i].dests &= !all_occ;
            all_possible_moves[i].dests = (((pawn_sq << 7) & !FILE_A) | ((pawn_sq << 9) & !FILE_H)) & (black_occ | self.enpassant_location);
            all_possible_moves[i].enpassant_location = self.enpassant_location;
        }
        all_possible_moves
    }
    
    fn generate_pseudolegal_black_pawn_moves(&self) -> [Move; 8] {
        let white_occ = self.get_combined_white_pieces();
        let all_occ = self.get_combined_black_pieces() | white_occ;
        let mut source = self.white_pawns;
        let mut all_possible_moves = [Move::new(0, 0, Colors::WHITE, 0); 8];
        for i in 0..8 {
            let (pawn_ind, pawn_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = pawn_sq;
            all_possible_moves[i].dests = (pawn_sq >> 8) | ((pawn_sq & RANK_7) >> 16);
            all_possible_moves[i].dests &= !all_occ;
            all_possible_moves[i].dests = (((pawn_sq >> 7) & !FILE_H) | ((pawn_sq >> 9) & !FILE_A)) & (white_occ | self.enpassant_location);
            all_possible_moves[i].enpassant_location = self.enpassant_location;
        }
        all_possible_moves
    }
    
    #[inline]
    fn generate_pseudolegal_rook_move_for_sq(sq_ind: usize, total_occ: u64, own_occ: u64) -> u64 {
        let mask = ROOK_MOVES_NO_RAY_ENDS[sq_ind];
        let needs_pext = ((mask & total_occ) != 0) as usize; // TODO wrong; mask needs to be not including the ends of rays 
        let pext = unsafe { _pext_u64(total_occ, mask) } as usize;
        let result = [ROOK_MOVES[sq_ind], ROOK_PEXT_TABLE[sq_ind][pext]];
        result[needs_pext] & !own_occ
    }
    
    // #[unroll_for_loops]
    pub fn generate_pseudolegal_rook_moves(&self, color: Colors) -> [Move; 10] {
        let cidx = color as usize;
        let all_pieces = [self.get_combined_white_pieces(), self.get_combined_black_pieces()];
        let own_occ = all_pieces[cidx];
        let other_occ = all_pieces[(cidx + 1) % 2];
        let total_occ = own_occ | other_occ;
        let mut source = [self.white_rooks, self.black_rooks][cidx];
        let mut all_possible_moves: [Move; 10] = [Move::new(0, 0, color, 0); 10];
        for i in 0..10 {
            let (rook_ind, rook_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = rook_sq; // impossible to have a zero source
            all_possible_moves[i].dests = Chessboard::generate_pseudolegal_rook_move_for_sq(rook_ind, total_occ, own_occ);
        }
        all_possible_moves
    }
    
    // #[unroll_for_loops]
    pub fn generate_pseudolegal_knight_moves(&self, color: Colors) -> [Move; 10] {
        let cidx = color as usize;
        let all_pieces = [self.get_combined_white_pieces(), self.get_combined_black_pieces()];
        let own_occ = all_pieces[cidx];
        let mut source = [self.white_knights, self.black_knights][cidx];
        let mut all_possible_moves: [Move; 10] = [Move::new(0, 0, color, 0); 10];
        for i in 0..10 {
            let (knight_ind, knight_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = knight_sq;
            all_possible_moves[i].dests = KNIGHT_MOVES[knight_ind];
            all_possible_moves[i].dests &= !own_occ;
        }
        all_possible_moves
    }
    
    #[inline]
    fn generate_pseudolegal_bishop_move_for_sq(sq_ind: usize, total_occ: u64, own_occ: u64) -> u64 {
        let mask = BISHOP_MOVES[sq_ind] & !BOARD_EDGES;
        let needs_pext = ((mask & total_occ) != 0) as usize;
        let pext = unsafe { _pext_u64(total_occ, mask) as usize};
        let result = [BISHOP_MOVES[sq_ind], BISHOP_PEXT_TABLE[sq_ind][pext]];
        result[needs_pext] & !own_occ
    }
    
    pub fn generate_pseudolegal_bishop_moves(&self, color: Colors) -> [Move; 10] {
        let cidx = color as usize;
        let all_pieces = [self.get_combined_white_pieces(), self.get_combined_black_pieces()];
        let own_occ = all_pieces[cidx];
        let other_occ = all_pieces[(cidx + 1) % 2];
        let total_occ = own_occ | other_occ;
        let mut source = [self.white_bishops, self.black_bishops][cidx];
        let mut all_possible_moves: [Move; 10] = [Move::new(0, 0, color, 0); 10];
        for i in 0..10 {
            let (bishop_ind, bishop_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = bishop_sq;
            all_possible_moves[i].dests = Chessboard::generate_pseudolegal_bishop_move_for_sq(bishop_ind, total_occ, own_occ);
        }
        all_possible_moves
    }

    pub fn generate_pseudolegal_king_moves(&self, color: Colors) -> u64 {
        let cidx = color as usize;
        let all_pieces = [self.get_combined_white_pieces(), self.get_combined_black_pieces()];
        let own_occ = all_pieces[cidx];
        let source = [self.white_king, self.black_king][cidx];
        let possible_moves = KING_MOVES[source.trailing_zeros() as usize];
        possible_moves & !own_occ
    }

    pub fn generate_pseudolegal_queen_moves(&self, color: Colors) -> [Move; 9] { 
        let cidx = color as usize;
        let all_pieces = [self.get_combined_white_pieces(), self.get_combined_black_pieces()];
        let own_occ = all_pieces[cidx];
        let other_occ = all_pieces[(cidx + 1) % 2];
        let total_occ = own_occ | other_occ;
        let mut source = [self.white_queens, self.black_queens][cidx];
        let mut all_possible_moves: [Move; 9] = [Move::new(0, 0, color, 0); 9];
        for i in 0..9 {
            let (queen_ind, queen_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].source = queen_sq;
            all_possible_moves[i].dests = Chessboard::generate_pseudolegal_bishop_move_for_sq(queen_ind, total_occ, own_occ) | Chessboard::generate_pseudolegal_rook_move_for_sq(queen_ind, total_occ, own_occ);
        }
        all_possible_moves
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    fn assert_white_rook_move_illegal_when_dest_occupied(source: u64, dest: u64) {
        let illegal_move = Move::new(source, dest, Colors::WHITE, 0);
        let mut test_board = Chessboard::new_blank();
        test_board.white_rooks = source | dest;
        let legal_moves = test_board.generate_pseudolegal_rook_moves(Colors::WHITE);
        for mv in legal_moves {
            assert!(!(mv.source == source && mv.dests == dest));
            assert!(!(mv.dests == source && mv.source == dest));
        }
        assert_eq!(legal_moves.iter().filter(|&&mv| mv.source != 0).collect::<Vec<&Move>>().len(), 2);
    }
    
    #[test]
    fn test_simple_illegal_rook_move() {
        for i in 0usize..64 {
            let source = 1u64 << i;
            let rank = get_rank_index(source).unwrap();
            if rank < 7 {
                let vstack = source << 8;
                assert_white_rook_move_illegal_when_dest_occupied(source, vstack);
            }
            if rank > 0 {
                let vstack = source >> 8;
                assert_white_rook_move_illegal_when_dest_occupied(source, vstack);
            }
            let file = get_file_index(source).unwrap();
            if file > 0 {
                let hstack = source << 1;
                assert_white_rook_move_illegal_when_dest_occupied(source, hstack);
            }
            if file < 7 {
                let hstack = source >> 1;
                assert_white_rook_move_illegal_when_dest_occupied(source, hstack);
            }
        }
    }
    
    fn check_occlusion_rooks(sq: u64, occluder: u64, occluded: u64) {
        let mut test_board = Chessboard::new_blank();
        test_board.white_rooks = sq;
        test_board.black_pawns = occluder | occluded;
        let mvs = test_board.generate_pseudolegal_rook_moves(Colors::WHITE);
        for mv in mvs {
            if mv.is_legal() {
                assert_eq!(occluded & mv.dests, 0);
            }
        }
    }
    
    #[test]
    fn test_rook_blockers_move() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for i in 0usize..64 {
            let source = 1u64 << i;
            let source_rank = get_rank_index(source).unwrap();
            let source_file = get_file_index(source).unwrap();
            for i in 0..10 {  // TODO generate a random row, shift it around, assert that the row above / below source is not hit by any legal move; repeat for files
                if source_rank < 6 {
                    let random_row: u64 = rng.random::<u64>() & RANK_1;
                    let random_row = (random_row << (8 * (source_rank + 1))) & !source;
                    let above_random_row = random_row << (8 * (source_rank + 2));
                    if random_row > 0 {
                        check_occlusion_rooks(source, random_row, above_random_row);
                    }
                }
                if source_file < 6 {
                    let random_file = rng.random::<u64>() & FILE_A;  // TODO a utility for moving things over file-wise
                    let random_file = shift_safe_files(random_file, source_file as i32) & !source;
                    let file_right = shift_safe_files(random_file, 1);
                    if random_file > 0 {
                        check_occlusion_rooks(source, random_file, file_right);
                    }
                }
            }
        }
    }
}
