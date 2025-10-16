use core::arch::x86_64::_pext_u64;
use chess_tables;
use chess_utils::consts::*;
use chess_utils::utils::*;
use gen_tables::*;

use crate::utils::*;


#[repr(usize)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Colors {
    WHITE = 0usize,
    BLACK = 1usize,
}

impl Colors {
    pub fn map_usize_to_color(n: usize) -> Colors {
        [Colors::WHITE, Colors::BLACK][n % 2]
    }

    pub fn opposite_color(color: Colors) -> Colors {
        Colors::map_usize_to_color(color as usize + 1)
    }
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

#[repr(usize)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PieceType {
    PAWN = 0,
    ROOK = 1,
    KNIGHT = 2,
    BISHOP = 3,
    QUEEN = 4,
    KING = 5,
    INVALID = 6,
}

static MAP_TO_PIECETYPE: [PieceType; 7] = [PieceType::PAWN, PieceType::ROOK, PieceType::KNIGHT, PieceType::BISHOP, PieceType::QUEEN, PieceType::KING, PieceType::INVALID];
impl PieceType {
    #[inline]
    pub fn from_usize(n: usize) -> PieceType {
        let ind = branchless_select(n > 6, n as u64, 6) as usize;
        MAP_TO_PIECETYPE[ind]
    }

    #[inline]
    pub fn from_u64(n: u64) -> PieceType {
        PieceType::from_usize(n as usize)
    }

    #[inline]
    pub fn to_u64(&self) -> u64 {
        (*self as usize) as u64
    }
}


#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Move {
    source: u64,
    dest: u64,
    piece_type: PieceType,
    color: Colors,
    enpassant_to: u64,
    rook_to_from: u64, // castling use only
    king_to: u64, //castling use only
    promotion: PieceType
} // TODO compress into single u64


impl Move {
    #[inline]
    pub fn new(source: u64, dest: u64, piece_type: PieceType, color: Colors, enpassant_to: u64) -> Move {
        Move { source, dest, piece_type, color, enpassant_to, rook_to_from: 0, king_to: 0 , promotion: PieceType::INVALID}
    }

    #[inline]
    pub fn new_invalid(color: Colors) -> Move {
        Move { source: 0, dest: 0, piece_type: PieceType::INVALID, color: color, enpassant_to: 0, rook_to_from: 0, king_to: 0, promotion: PieceType::INVALID }
    }

    #[inline]
    pub fn new_kingside_castle(color: Colors) -> Move {
        let kingside_king_to = 0b00000010u64;
        let kingside_rook_to_from = 0b00000101u64;
        let king_to = [kingside_king_to, kingside_king_to << 56][color as usize];
        let rook_to_from = [kingside_rook_to_from, kingside_rook_to_from << 56][color as usize];
        Move { source: 0u64, dest: 0u64, piece_type: PieceType::KING, color: Colors::WHITE, enpassant_to: 0, rook_to_from, king_to, promotion: PieceType::INVALID }
    }

    #[inline]
    pub fn new_queenside_castle(color: Colors) -> Move {
        let queenside_rook_to_from = 0b10010000u64;
        let queenside_king_to = 0b00100000u64;
        let king_to = [queenside_king_to, queenside_king_to << 56][color as usize];
        let rook_to_from = [queenside_rook_to_from, queenside_rook_to_from << 56][color as usize];
        Move { source: 0u64, dest: 0u64, piece_type: PieceType::KING, color: Colors::WHITE, enpassant_to: 0, rook_to_from, king_to, promotion: PieceType::INVALID }
    }

    #[inline]
    pub fn is_kingside_castle(&self) -> bool {
        let kingside_king_to = 0b00000010u64;
        let kingside_rook_to = 0b00000100u64;
        (self.rook_to_from == kingside_rook_to || self.rook_to_from == (kingside_rook_to << 56))
        && (self.king_to == kingside_king_to || self.king_to == (kingside_king_to << 56))
    }

    #[inline]
    pub fn is_queenside_castle(&self) -> bool {
        let queenside_rook_to = 0b00010000u64;
        let queenside_king_to = 0b00100000u64;
        (self.rook_to_from == queenside_rook_to || self.rook_to_from == (queenside_rook_to << 56))
        && (self.king_to == queenside_king_to || self.king_to == (queenside_king_to << 56))
    }
    
    #[inline]
    pub fn is_castling(&self) -> bool {
        self.is_kingside_castle() || self.is_queenside_castle()
    }

    #[inline]
    pub fn is_pawn_move_2sq(&self) -> bool {
        let rank_4 = RANK_2 << 16;
        let rank_5 = RANK_7 >> 16;
        let white_cond = ((self.source & RANK_2) > 0) && ((self.dest & rank_4) > 0);
        let black_cond = ((self.source & RANK_7) > 0) && ((self.dest & rank_5) > 0);
        (self.piece_type == PieceType::PAWN) && (white_cond || black_cond)
    }

    #[inline]
    pub fn is_enpassant(&self) -> bool {
        self.enpassant_to != 0
    }

    #[inline]
    pub fn enpassant_takes_sq(&self) -> u64 {
        branchless_select(self.is_enpassant(), 0u64, branchless_select(self.color == Colors::WHITE, self.enpassant_to << 8, self.enpassant_to >> 8))
    }
}


#[derive(Clone, Copy, PartialEq, Debug)]
struct PossibleMoves {
    source: u64,
    dests: u64,
    color: Colors,
    enpassant_takes: u64,
    piece_type: PieceType,
} // TODO can compress this a lot, probably into one u64 or u32, use the move index 0-64


impl PossibleMoves {  // TODO just remove this intermediary struct alltogether
    #[inline]
    pub fn new(source: u64, dests: u64, color: Colors, enpassant_takes: u64, piece_type: PieceType) -> PossibleMoves {
        PossibleMoves {
            source,
            dests,
            color,
            enpassant_takes,
            piece_type,
        }
    }

    #[inline]
    pub fn exists(&self) -> bool {
        self.source != 0
    }

    #[inline]
    fn enpassant_opportunity(&self, dest: u64) -> u64 {
        let rank_4 = RANK_2 << 16;
        let rank_5 = RANK_7 >> 16;
        let white_is_taking_pos = (rank_5 & self.enpassant_takes) << 8;
        let black_is_taking_pos = (rank_4 & self.enpassant_takes) >> 8;
        let result = branchless_select(self.color == Colors::BLACK, white_is_taking_pos, black_is_taking_pos);
        dest & result
    }

    #[inline]
    fn could_promote(&self, dest: u64) -> bool {
        let white_cond = (dest & RANK_8) > 0;
        let black_cond = (dest & RANK_1) > 0;
        (self.piece_type == PieceType::PAWN) && (branchless_select(self.color == Colors::WHITE, black_cond as u64, white_cond as u64) == 1)
    }
    // TODO this is not branchless
    #[inline]
    pub fn to_moves(&self) -> ([Move; 64], usize) { // TODO handle promotions here
        let mut dests = self.dests;
        let mut result: [Move; 64] = [Move::new(self.source, 0u64, PieceType::INVALID, self.color, self.enpassant_takes); 64]; // TODO can definitely avoid some construction here
        let mut i = 0;
        while dests != 0 && self.source != 0 {
            let nxt = 1u64 << dests.trailing_zeros();
            let can_promote = self.could_promote(nxt);
            let increment = branchless_select(can_promote, 1, 4) as usize;
            for j in 0..increment {
                dests &= !nxt;
                result[i].dest = nxt;
                result[i].piece_type = self.piece_type;
                result[i].enpassant_to = self.enpassant_opportunity(nxt);
                result[i].promotion = PieceType::from_usize(branchless_select(can_promote, (PieceType::INVALID as usize) as u64, (j as u64) + 1u64) as usize);
                i += 1;
            }
        }
        (result, i)
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

#[derive(Clone, Copy)]
pub struct Chessboard {
    // TODO refactor this for [color, piece type] index
    pieces: [[u64; 6]; 2],
    enpassant_takes: u64, // bitboard square that gets set whe na pawn moves 2 squares forwards
    kingside_castling_rights: [bool; 2],
    queenside_castling_rights: [bool; 2],
    moves_since_takes: i32, // 50 move rule
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

        let white = [
            white_pawns,
            white_rooks,
            white_knights,
            white_bishops,
            white_queens,
            white_king,
        ];
        let black = [
            black_pawns,
            black_rooks,
            black_knights,
            black_bishops,
            black_queens,
            black_king,
        ];

        Chessboard {
            pieces: [white, black],
            enpassant_takes: 0,
            kingside_castling_rights: [true, true],
            queenside_castling_rights: [true, true],
            moves_since_takes: 0,
        }
    }

    pub fn new_blank() -> Chessboard {
        Chessboard {
            pieces: [[0u64; 6]; 2],
            enpassant_takes: 0,
            kingside_castling_rights: [false, false],
            queenside_castling_rights: [false, false],
            moves_since_takes: 0,
        }
    }

    #[inline(always)]
    pub fn get_piece(&self, color: Colors, piece: PieceType) -> u64 {
        self.pieces[color as usize][piece as usize]
    }

    #[inline(always)]
    pub fn set_piece(&mut self, color: Colors, piece: PieceType, board: u64) {
        self.pieces[color as usize][piece as usize] = board;
    }

    #[inline]
    fn get_combined_pieces(&self, color: Colors) -> u64 {
        self.get_piece(color, PieceType::ROOK)
            | self.get_piece(color, PieceType::KNIGHT)
            | self.get_piece(color, PieceType::BISHOP)
            | self.get_piece(color, PieceType::QUEEN)
            | self.get_piece(color, PieceType::KING)
            | self.get_piece(color, PieceType::PAWN)
    }

    #[inline]
    fn get_full_pieces_mask(&self) -> u64 {
        self.get_combined_pieces(Colors::WHITE) | self.get_combined_pieces(Colors::BLACK)
    }

    fn generate_pseudolegal_white_pawn_moves(&self) -> [PossibleMoves; 8] {
        // TODO this is enough to justify a pregen table, will also allow removal of branch later that requires checking if the color is white or black, by indexing into the table with the color
        let black_occ = self.get_combined_pieces(Colors::BLACK);
        let all_occ = self.get_combined_pieces(Colors::WHITE) | black_occ;
        let mut source = self.get_piece(Colors::WHITE, PieceType::PAWN);
        let mut all_possible_moves = [PossibleMoves::new(0, 0, Colors::WHITE, 0, PieceType::PAWN); 8];
        for i in 0..8 {
            let (pawn_ind, pawn_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::PAWN, PieceType::INVALID][(pawn_sq == 0) as usize];
            all_possible_moves[i].source = pawn_sq;
            all_possible_moves[i].dests = (pawn_sq << 8) & !all_occ;
            all_possible_moves[i].dests |= (((pawn_sq & RANK_2) << 8) & !all_occ) << 8;
            all_possible_moves[i].dests &= !all_occ;
            all_possible_moves[i].dests |= (((pawn_sq << 7) & !FILE_A)
                | ((pawn_sq << 9) & !FILE_H))
                & (black_occ | (self.enpassant_takes << 8));
            all_possible_moves[i].enpassant_takes = self.enpassant_takes;
        }
        all_possible_moves
    }

    fn generate_pseudolegal_black_pawn_moves(&self) -> [PossibleMoves; 8] { // TODO handle promotions, fix en passant
        let white_occ = self.get_combined_pieces(Colors::WHITE);
        let all_occ = self.get_combined_pieces(Colors::BLACK) | white_occ;
        let mut source = self.get_piece(Colors::BLACK, PieceType::PAWN);
        let mut all_possible_moves = [PossibleMoves::new(0, 0, Colors::BLACK, 0, PieceType::PAWN); 8];
        for i in 0..8 {
            let (pawn_ind, pawn_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::PAWN, PieceType::INVALID][(pawn_sq == 0) as usize];
            all_possible_moves[i].source = pawn_sq;
            all_possible_moves[i].dests = (pawn_sq >> 8) & !all_occ;
            all_possible_moves[i].dests |= (((pawn_sq & RANK_7) >> 8) & !all_occ) >> 8;
            all_possible_moves[i].dests &= !all_occ;
            all_possible_moves[i].dests |= (((pawn_sq >> 7) & !FILE_H)
                | ((pawn_sq >> 9) & !FILE_A))
                & (white_occ | (self.enpassant_takes >> 8));
            all_possible_moves[i].enpassant_takes = self.enpassant_takes;
        }
        all_possible_moves
    }

    #[inline]
    fn generate_pseudolegal_rook_move_for_sq(sq_ind: usize, total_occ: u64, own_occ: u64) -> u64 {
        let mask = ROOK_MOVES_NO_RAY_ENDS[sq_ind];
        let needs_pext = ((mask & total_occ) != 0) as usize;
        let pext = unsafe { _pext_u64(total_occ, mask) } as usize;
        let result = [ROOK_MOVES[sq_ind], ROOK_PEXT_TABLE[sq_ind][pext]];  // TODO when pext is zero, should just be ROOK_MOVES
        result[needs_pext] & !own_occ
    }

    // #[unroll_for_loops]
    pub fn generate_pseudolegal_rook_moves(&self, color: Colors) -> [PossibleMoves; 10] {
        let other_color = Colors::opposite_color(color);
        let own_occ = self.get_combined_pieces(color);
        let other_occ = self.get_combined_pieces(other_color);
        let total_occ = own_occ | other_occ;
        let mut source = self.get_piece(color, PieceType::ROOK);
        let mut all_possible_moves: [PossibleMoves; 10] = [PossibleMoves::new(0, 0, color, 0, PieceType::ROOK); 10];
        for i in 0..10 {
            let (rook_ind, rook_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::ROOK, PieceType::INVALID][(rook_sq == 0) as usize];
            all_possible_moves[i].source = rook_sq; // impossible to have a zero source
            all_possible_moves[i].dests =
                Chessboard::generate_pseudolegal_rook_move_for_sq(rook_ind, total_occ, own_occ);
        }
        all_possible_moves
    }

    // #[unroll_for_loops]
    pub fn generate_pseudolegal_knight_moves(&self, color: Colors) -> [PossibleMoves; 10] {
        let own_occ = self.get_combined_pieces(color);
        let mut source = self.get_piece(color, PieceType::KNIGHT);
        let mut all_possible_moves: [PossibleMoves; 10] = [PossibleMoves::new(0, 0, color, 0, PieceType::KNIGHT); 10];
        for i in 0..10 {
            let (knight_ind, knight_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::KNIGHT, PieceType::INVALID][(knight_sq == 0) as usize];
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
        let pext = unsafe { _pext_u64(total_occ, mask) as usize };
        let result = [BISHOP_MOVES[sq_ind], BISHOP_PEXT_TABLE[sq_ind][pext]];
        result[needs_pext] & !own_occ
    }

    pub fn generate_pseudolegal_bishop_moves(&self, color: Colors) -> [PossibleMoves; 10] {
        let other_color = Colors::opposite_color(color);
        let own_occ = self.get_combined_pieces(color);
        let other_occ = self.get_combined_pieces(other_color);
        let total_occ = own_occ | other_occ;
        let mut source = self.get_piece(color, PieceType::BISHOP);
        let mut all_possible_moves: [PossibleMoves; 10] = [PossibleMoves::new(0, 0, color, 0, PieceType::BISHOP); 10];
        for i in 0..10 {
            let (bishop_ind, bishop_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::BISHOP, PieceType::INVALID][(bishop_sq == 0) as usize];
            all_possible_moves[i].source = bishop_sq;
            all_possible_moves[i].dests =
                Chessboard::generate_pseudolegal_bishop_move_for_sq(bishop_ind, total_occ, own_occ);
        }
        all_possible_moves
    }

    pub fn generate_pseudolegal_king_moves(&self, color: Colors) -> ([Move; 8], usize) {
        let mut ret: [Move; 8] = [Move::new_invalid(color); 8];
        let mut count_legal = 0usize;

        let kingside = self.may_castle_kingside(color);
        let kingside_mv = Move::new_kingside_castle(color);
        ret[count_legal].source = branchless_select(kingside, 0, branchless_select(color == Colors::WHITE, 0b00001000 << 56, 0b00001000));
        ret[count_legal].dest = branchless_select(kingside, ret[count_legal].dest, kingside_mv.dest);
        ret[count_legal].king_to = branchless_select(kingside, ret[count_legal].king_to, kingside_mv.king_to);
        ret[count_legal].rook_to_from = branchless_select(kingside, ret[count_legal].rook_to_from, kingside_mv.rook_to_from);
        ret[count_legal].piece_type = PieceType::from_u64(branchless_select(kingside, ret[count_legal].piece_type.to_u64() , PieceType::KING.to_u64()));
        count_legal += branchless_select(kingside, 0, 1) as usize;

        let queenside = self.may_castle_queenside(color);
        let queenside_mv = Move::new_queenside_castle(color);
        ret[count_legal].source = branchless_select(queenside, 0, branchless_select(color == Colors::WHITE, 0b00001000 << 56, 0b00001000));
        ret[count_legal].dest = branchless_select(queenside, ret[count_legal].dest, 0u64);
        ret[count_legal].king_to = branchless_select(queenside, ret[count_legal].king_to, queenside_mv.king_to);
        ret[count_legal].rook_to_from = branchless_select(queenside, ret[count_legal].rook_to_from, queenside_mv.rook_to_from);
        ret[count_legal].piece_type = PieceType::from_u64(branchless_select(queenside, ret[count_legal].piece_type.to_u64() , PieceType::KING.to_u64()));
        count_legal += branchless_select(queenside, 0, 1) as usize;

        let own_occ = self.get_combined_pieces(color);
        let source = self.get_piece(color, PieceType::KING);
        let mut dests = KING_MOVES[source.trailing_zeros() as usize] & !own_occ;
        while dests != 0 {
            let nxt = 1u64 << dests.trailing_zeros();
            dests &= !nxt;
            ret[count_legal].source = source;
            ret[count_legal].dest = nxt;
            ret[count_legal].piece_type = PieceType::KING;
            count_legal += 1;
        }

        (ret, count_legal)
    }

    pub fn generate_pseudolegal_queen_moves(&self, color: Colors) -> [PossibleMoves; 9] {
        let other_color = Colors::opposite_color(color);
        let own_occ = self.get_combined_pieces(color);
        let other_occ = self.get_combined_pieces(other_color);
        let total_occ = own_occ | other_occ;
        let mut source = self.get_piece(color, PieceType::QUEEN);
        let mut all_possible_moves: [PossibleMoves; 9] = [PossibleMoves::new(0, 0, color, 0, PieceType::QUEEN); 9];
        for i in 0..9 {
            let (queen_ind, queen_sq) = get_next_piece_as_u64_and_rm_from_source(&mut source);
            all_possible_moves[i].piece_type = [PieceType::QUEEN, PieceType::INVALID][(queen_sq == 0) as usize];
            all_possible_moves[i].source = queen_sq;
            all_possible_moves[i].dests =
                Chessboard::generate_pseudolegal_bishop_move_for_sq(queen_ind, total_occ, own_occ)
                    | Chessboard::generate_pseudolegal_rook_move_for_sq(
                        queen_ind, total_occ, own_occ,
                    );
        }
        all_possible_moves
    }

    // TODO this could be slow
    #[inline(always)]
    pub fn play_move(&self, mv: &Move) -> Chessboard {
        debug_assert!(mv.source.count_ones() == 1 || mv.is_queenside_castle() || mv.is_kingside_castle());
        debug_assert!(mv.dest.count_ones() == 1 || mv.is_queenside_castle() || mv.is_kingside_castle());

        let mut nxt_board = self.clone();
        // remove current piece
        nxt_board.pieces[mv.color as usize][mv.piece_type as usize] &= !mv.source;

        let other_color = Colors::opposite_color(mv.color);

        // calculate takes
        for pt in 0usize..6 { // does nothing if enpassant or castling
            nxt_board.pieces[other_color as usize][pt] &= !mv.dest; // if enpassant or castling, this isn't set
        }
        nxt_board.moves_since_takes += 1 - ((self.get_combined_pieces(other_color).count_ones() as i32) - (nxt_board.get_combined_pieces(other_color).count_ones() as i32));
        // move piece / promotion handling
        let piece_type_ind = branchless_select(mv.promotion != PieceType::INVALID, mv.piece_type as u64, mv.promotion as u64) as usize;
        nxt_board.pieces[mv.color as usize][piece_type_ind] |= mv.dest;

        nxt_board.pieces[other_color as usize][PieceType::PAWN as usize] &= !mv.enpassant_takes_sq();
        nxt_board.enpassant_takes = branchless_select(mv.is_pawn_move_2sq(), 0u64, mv.dest);

        // Handle castling
        // for castling, the above does nothing. for not castling, the below does nothing.
        let color_bool = mv.color == Colors::BLACK;
        let cur_rooks = nxt_board.get_piece(mv.color, PieceType::ROOK);
        let kingside_rook_from_to = branchless_select(color_bool, 0b00000101u64 ^ cur_rooks, (0b00000101u64 << 56) ^ cur_rooks);  // TODO verify that rook square is empty in castle legality check
        let queenside_rook_from_to = branchless_select(color_bool, 0b10010000u64 ^ cur_rooks, (0b10010000u64 << 56) ^ cur_rooks);
        
        nxt_board.set_piece(mv.color, PieceType::KING, branchless_select(mv.is_castling(), nxt_board.get_piece(mv.color, PieceType::KING), mv.king_to));
        nxt_board.set_piece(mv.color, PieceType::ROOK, branchless_select(
            mv.is_queenside_castle(),
            branchless_select(
                mv.is_kingside_castle(),
                nxt_board.get_piece(mv.color, PieceType::ROOK),
                kingside_rook_from_to,
            ),
            queenside_rook_from_to,
        ));
        // Update castling rights
        let king_start = branchless_select(mv.color == Colors::WHITE, 0b00001000 << 56, 0b00001000);
        let kingside_rook_start = branchless_select(mv.color == Colors::WHITE, 0b1 << 56, 0b1);
        let kingside_rights = self.kingside_castling_rights[mv.color as usize];
        let queenside_rook_start = branchless_select(mv.color == Colors::WHITE, 0b10000000 << 56, 0b10000000);
        let queenside_rights = self.queenside_castling_rights[mv.color as usize];

        let kingside_condition = kingside_rights && (((mv.piece_type == PieceType::ROOK) && ((mv.source & kingside_rook_start) > 0)) || ((mv.piece_type == PieceType::KING) && ((mv.source & king_start) > 0)));
        let queenside_condition = queenside_rights && (((mv.piece_type == PieceType::ROOK) && ((mv.source & queenside_rook_start) > 0)) || ((mv.piece_type == PieceType::KING) && ((mv.source & king_start) > 0)));

        nxt_board.kingside_castling_rights[mv.color as usize] = branchless_select(kingside_condition, kingside_rights as u64, false as u64) != 0;
        nxt_board.queenside_castling_rights[mv.color as usize] = branchless_select(queenside_condition, queenside_rights as u64, false as u64) != 0; // TODO write test to verify this updates correctly

        nxt_board
    }

    #[inline]
    fn generate_union_move_dests_for_color_nopawns(&self, color: Colors) -> u64 {  // TODO this is pretty slow, unfortunately necessary for castle checking
        let rook_dests: u64 = self.generate_pseudolegal_rook_moves(color).iter().fold(0u64, |x, y| { x | y.dests });
        let knight_dests: u64 = self.generate_pseudolegal_knight_moves(color).iter().fold(0u64, |x, y| { x | y.dests });
        let bishop_dests: u64 = self.generate_pseudolegal_bishop_moves(color).iter().fold(0u64, |x, y| { x | y.dests });
        let queen_dests: u64 = self.generate_pseudolegal_queen_moves(color).iter().fold(0u64, |x, y| { x | y.dests });
        rook_dests | knight_dests | bishop_dests | queen_dests
    }

    #[inline]
    fn generate_union_move_dests_for_color(&self, color: Colors) -> u64 { // TODO probably faster to branch here tbh, but even better to just rewrite all the fucking code so that this isn't the interface
        // TODO maybe worth even having a separate generator that generates just the sum OR of the dests for each piece, specifically for castle checking
        let pawn_dests: u64 = [self.generate_pseudolegal_white_pawn_moves(), self.generate_pseudolegal_black_pawn_moves()][color as usize].iter().fold(0u64, |x, y| { x | y.dests });
        pawn_dests | self.generate_union_move_dests_for_color_nopawns(color)
    }
    
    fn set_array_while_valid(moves: &mut [Move; 256], possible_moves: &[PossibleMoves], start: &mut usize) {
        for pmv in possible_moves {
            let (mvs, count) = pmv.to_moves();
            for i in 0..count {
                moves[*start] = mvs[i];
                *start += 1;
            }
        }
    }

    fn set_array_while_valid_mvs(moves: &mut [Move; 256], set_with: &[Move], start: &mut usize, length: usize) {
        let mut length = length;
        for i in 0..length {
            moves[*start + i] = set_with[i];
        }
        *start += length;
    }

    pub fn generate_all_moves(&self, color: Colors) -> ([Move; 256], usize) {
        // TODO this is wasteful, and can definitely be improved, although the compiler may save us here.
        let pawn_moves = if color == Colors::WHITE { self.generate_pseudolegal_white_pawn_moves() } else { self.generate_pseudolegal_black_pawn_moves() };
        let rook_moves = self.generate_pseudolegal_rook_moves(color);
        let knight_moves = self.generate_pseudolegal_knight_moves(color);
        let bishop_moves = self.generate_pseudolegal_bishop_moves(color);
        let queen_moves = self.generate_pseudolegal_queen_moves(color);
        let (king_move, valid_king_moves) = self.generate_pseudolegal_king_moves(color);
        let mut decomposed_moves: [Move; 256] = [Move::new(0, 0, PieceType::INVALID, color, self.enpassant_takes); 256];
        let mut i = 0usize;
        Self::set_array_while_valid(&mut decomposed_moves, &pawn_moves, &mut i);
        Self::set_array_while_valid(&mut decomposed_moves, &rook_moves, &mut i);
        Self::set_array_while_valid(&mut decomposed_moves, &knight_moves, &mut i);
        Self::set_array_while_valid(&mut decomposed_moves, &bishop_moves, &mut i);
        Self::set_array_while_valid(&mut decomposed_moves, &queen_moves, &mut i);
        Self::set_array_while_valid_mvs(&mut decomposed_moves, &king_move, &mut i, valid_king_moves);
        (decomposed_moves, i)
    }

    #[inline]
    pub fn is_king_in_check(&self, color: Colors) -> bool {
        let other_color = Colors::opposite_color(color);
        let king_loc = self.get_piece(color, PieceType::KING);
        let king_ind = king_loc.trailing_zeros() as usize;
        let own_occ = self.get_combined_pieces(color);
        let total_occ = self.get_combined_pieces(other_color) | own_occ;
        let other_knights = self.get_piece(other_color, PieceType::KNIGHT);
        let other_bishops = self.get_piece(other_color, PieceType::BISHOP);
        let other_rooks = self.get_piece(other_color, PieceType::ROOK);
        let other_queens = self.get_piece(other_color, PieceType::QUEEN);
        let knight_sqs = KNIGHT_MOVES[king_ind];
        let bishop_sqs = Chessboard::generate_pseudolegal_bishop_move_for_sq(king_ind, total_occ, own_occ);
        let rook_sqs = Chessboard::generate_pseudolegal_rook_move_for_sq(king_ind, total_occ, own_occ);
        let white_pawn_intersect = ((((king_loc >> 7) & !FILE_H) | ((king_loc >> 9) & !FILE_A)) & self.get_piece(Colors::WHITE, PieceType::PAWN)) > 0;
        let black_pawn_intersect = ((((king_loc << 7) & !FILE_A) | ((king_loc << 9) & !FILE_H)) & self.get_piece(Colors::BLACK, PieceType::PAWN)) > 0;
        let other_king_exists = self.get_piece(other_color, PieceType::KING) != 0; // TODO stinky
        let other_king_sqs = branchless_select(other_king_exists, 0, KING_MOVES[branchless_select(other_king_exists, 0, self.get_piece(other_color, PieceType::KING).trailing_zeros() as u64) as usize]);
        let attacked_by_pawn = [black_pawn_intersect, white_pawn_intersect][color as usize]; // white king is attacked by black pawns from above and vice versa
        (knight_sqs & other_knights > 0) || (bishop_sqs & other_bishops > 0) || (rook_sqs & other_rooks > 0) || ((bishop_sqs | rook_sqs) & other_queens > 0) || attacked_by_pawn || (king_loc & other_king_sqs > 0)
    }
    
    fn all_moves_lead_to_check(&self, color: Colors) -> bool { // TODO can make branchless?
        let (all_moves, num_legal_moves) = self.generate_all_moves(color);
        let mut every_move_leaves_king_in_check = true;
        for i in 0..num_legal_moves {
            let nxt_board = self.play_move(&all_moves[i]);
            every_move_leaves_king_in_check &= nxt_board.is_king_in_check(color);
        }
        every_move_leaves_king_in_check
    }

    pub fn is_checkmate(&self, color: Colors) -> bool {  // TODO tests
        // TODO this is another bloated operation that can be streamlined, probably
        let king_in_check = self.is_king_in_check(color);
        if king_in_check {
            return self.all_moves_lead_to_check(color);
        }
        false
    }
    
    pub fn is_stalemate(&self, color: Colors) -> bool {
        if self.is_king_in_check(color) {
            return false;
        }
        self.all_moves_lead_to_check(color)
    }
    
    pub fn is_initialized(&self) -> bool {
        (self.get_piece(Colors::BLACK, PieceType::KING) | self.get_piece(Colors::WHITE, PieceType::KING)).count_ones() == 2
    }

    // TODO write a MoveArray / ChessboardArray impl
    pub fn generate_next_legal_boards(&self, color: Colors) -> ([(Chessboard, Move); 256], usize) {
        // TODO optimize
        let (mvs, num_legal) = self.generate_all_moves(color);
        let mut ret: [(Chessboard, Move); 256] = [(self.clone(), Move::new_invalid(color)); 256];
        for i in 0..num_legal {
            ret[i].0 = ret[i].0.play_move(&mvs[i]);
            ret[i].1 = mvs[i];
        }
        (ret, num_legal)
    }
    // TODO insufficient mating material draw condition
    // TODO castling tests, play castling move

    #[inline]  // TODO check pawn in checkmate
    fn castling_helper(&self, color: Colors, inbetween_mask: u64) -> bool { // TODO this should just use pext and check the each square on the route
        let inbetween_space = branchless_select(color == Colors::BLACK, inbetween_mask, inbetween_mask << 56);
        let no_blockers = (inbetween_space & self.get_full_pieces_mask()) == 0;
        let other_color = Colors::opposite_color(color);
        let own_occ = self.get_combined_pieces(color);
        let other_occ = self.get_combined_pieces(other_color);
        let total_occ = own_occ | other_occ;
        let mut b = inbetween_mask;
        let mut is_not_attacked_enroute = true;
        while b != 0 {
            let sq_ind = b.trailing_zeros() as usize;
            let nxt = 1u64 << sq_ind;
            b &= !nxt;
            let bishop_rays = Self::generate_pseudolegal_bishop_move_for_sq(sq_ind, total_occ, own_occ);
            let rook_rays = Self::generate_pseudolegal_rook_move_for_sq(sq_ind, total_occ, own_occ);
            let knight_rays = KNIGHT_MOVES[sq_ind];
            is_not_attacked_enroute &= (bishop_rays & self.get_piece(other_color, PieceType::BISHOP) == 0) && (bishop_rays & self.get_piece(other_color, PieceType::QUEEN) == 0) && (rook_rays & self.get_piece(other_color, PieceType::ROOK) == 0) && (rook_rays & self.get_piece(other_color, PieceType::QUEEN) == 0) && (knight_rays & self.get_piece(other_color, PieceType::KNIGHT) == 0);
        }
        let pawn_diagonals = branchless_select(color == Colors::BLACK, (inbetween_mask << 7) | (inbetween_mask << 9), ((inbetween_mask << 56) >> 7) | ((inbetween_mask << 56) >> 9));
        let pawns_diagonal_from_path = self.get_piece(other_color, PieceType::PAWN) & pawn_diagonals;
        let not_attacked_along_path = is_not_attacked_enroute && (pawns_diagonal_from_path == 0);
        !self.is_king_in_check(color) && no_blockers && not_attacked_along_path
    }

    #[inline]
    pub fn may_castle_kingside(&self, color: Colors) -> bool { // TODO check attacks on king path, probably need pext tables in long term
        let color_can_castle = self.kingside_castling_rights[color as usize];  // TODO kingside rook move sets rights to false
        color_can_castle && self.castling_helper(color, 0b00000110u64)
    }

    #[inline]
    pub fn may_castle_queenside(&self, color: Colors) -> bool { // TODO check attacks on king path, probably need pext tables in long term
        let color_can_castle = self.queenside_castling_rights[color as usize];
        color_can_castle && self.castling_helper(color, 0b01110000u64)
    }

    #[inline]
    pub fn is_draw(&self) -> bool {  // TODO doesn't capture 3-fold repetition, this should be handled by the parent context
        let white_lone_king = self.get_combined_pieces(Colors::WHITE) == self.get_piece(Colors::WHITE, PieceType::KING);
        let black_lone_king = self.get_combined_pieces(Colors::BLACK) == self.get_piece(Colors::BLACK, PieceType::KING);
        let white_king_bishop = self.get_combined_pieces(Colors::WHITE) == (self.get_piece(Colors::WHITE, PieceType::BISHOP) | self.get_piece(Colors::WHITE, PieceType::KING)) && self.get_piece(Colors::WHITE, PieceType::BISHOP).count_ones() == 1;
        let black_king_bishop = self.get_combined_pieces(Colors::BLACK) == (self.get_piece(Colors::BLACK, PieceType::BISHOP) | self.get_piece(Colors::BLACK, PieceType::KING)) && self.get_piece(Colors::BLACK, PieceType::BISHOP).count_ones() == 1;
        let white_king_knight = self.get_combined_pieces(Colors::WHITE) == (self.get_piece(Colors::WHITE, PieceType::KNIGHT) | self.get_piece(Colors::WHITE, PieceType::KING)) && self.get_piece(Colors::WHITE, PieceType::KNIGHT).count_ones() == 1;
        let black_king_knight = self.get_combined_pieces(Colors::BLACK) == (self.get_piece(Colors::BLACK, PieceType::KNIGHT) | self.get_piece(Colors::BLACK, PieceType::KING)) && self.get_piece(Colors::BLACK, PieceType::KNIGHT).count_ones() == 1;

        white_lone_king && black_lone_king
            || white_lone_king && black_king_bishop
            || white_lone_king && black_king_knight
            || black_lone_king && white_king_bishop
            || black_lone_king && white_king_knight
            || (white_king_bishop && black_king_bishop)
            || (white_king_knight && black_king_knight)
            || (white_king_bishop && black_king_knight)
            || (white_king_knight && black_king_bishop)
            || self.moves_since_takes == 50
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    fn assert_white_rook_move_illegal_when_dest_occupied(source: u64, dest: u64) {
        let illegal_move = PossibleMoves::new(source, dest, Colors::WHITE, 0, PieceType::ROOK);
        let mut test_board = Chessboard::new_blank();
        test_board.set_piece(Colors::WHITE, PieceType::ROOK, source | dest);
        let legal_moves = test_board.generate_pseudolegal_rook_moves(Colors::WHITE);
        for mv in legal_moves {
            assert!(!(mv.source == source && mv.dests == dest));
            assert!(!(mv.dests == source && mv.source == dest));
        }
        assert_eq!(
            legal_moves
                .iter()
                .filter(|&&mv| mv.source != 0)
                .collect::<Vec<&PossibleMoves>>()
                .len(),
            2
        );
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
        test_board.set_piece(Colors::WHITE, PieceType::ROOK, sq);
        test_board.set_piece(Colors::BLACK, PieceType::PAWN, occluder | occluded);
        let mvs = test_board.generate_pseudolegal_rook_moves(Colors::WHITE);
        for mv in mvs {
            if mv.exists() {
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
            for i in 0..10 {
                if source_rank < 6 {
                    let random_row: u64 = rng.random::<u64>() & RANK_1;
                    let random_row = (random_row << (8 * (source_rank + 1))) & !source;
                    let above_random_row = random_row << (8 * (source_rank + 2));
                    if random_row > 0 {
                        check_occlusion_rooks(source, random_row, above_random_row);
                    }
                }
                if source_file < 6 {
                    let random_file = rng.random::<u64>() & FILE_A;
                    let random_file = shift_safe_files(random_file, source_file as i32) & !source;
                    let file_right = shift_safe_files(random_file, 1);
                    if random_file > 0 {
                        check_occlusion_rooks(source, random_file, file_right);
                    }
                }
            }
        }
    }

    fn mask_rect_excl_sq(sq: u64) -> (u64, u64, u64, u64) {
        let sq_file = get_file_index(sq).unwrap();
        let sq_rank = get_rank_index(sq).unwrap();

        let mut nw_block = 0u64;
        let mut ne_block = 0u64;
        let mut se_block = 0u64;
        let mut sw_block = 0u64;
        for i in 0..64 {
            let sq = 1u64 << i;
            let rank = get_rank_index(sq).unwrap();
            let file = get_file_index(sq).unwrap();
            if rank > sq_rank && file < sq_file {
                nw_block |= sq;
            } else if rank > sq_rank && file > sq_file {
                ne_block |= sq;
            } else if rank < sq_rank && file < sq_file {
                sw_block |= sq;
            } else if rank < sq_rank && file > sq_file {
                se_block |= sq;
            }
        }
        (nw_block, ne_block, se_block, sw_block)
    }

    fn get_closest_to_sq(board: u64, sq: u64) -> Option<u64> {
        let sq_rank = get_rank_index(sq).unwrap() as i32;
        let sq_file = get_file_index(sq).unwrap() as i32;
        let all_coordinates = collect_coordinates(board);
        let mut least_dist = i32::MAX;
        let mut closest_sq = 0u64;
        for (rank, file) in all_coordinates {
            let ranki = rank as i32;
            let filei = file as i32;
            let dist = (ranki - sq_rank).pow(2) + (filei - sq_file).pow(2);
            if dist < least_dist {
                closest_sq = map_rank_and_file_to_sq(rank, file);
                least_dist = dist;
            }
        }
        if closest_sq != 0 {
            Some(closest_sq)
        } else {
            None
        }
    }

    fn assert_closest_bounds_ray(sq: u64, ray: u64) {
        if ray == 0 {
            return;
        }
        let closest = get_closest_to_sq(ray, sq);
        match closest {
            Some(closest_sq) => {
                let least_dist = sq_dist(closest_sq, sq);
                for other_sq in get_all_individual_sq(ray) {
                    assert!(other_sq == closest_sq || sq_dist(other_sq, sq) > least_dist);
                }
            }
            None => {}
        }
    }

    #[test]
    fn test_bishop_blockers_move() {
        // TODO mask NW, NE, SE, SW quadrants to get the rays
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for i in 0usize..64 {
            let source = 1u64 << i;
            let mask = BISHOP_MOVES[i];
            for i in 0..10 {
                let random_mask = rng.random::<u64>() & mask;
                let (nw_block, ne_block, se_block, sw_block) = mask_rect_excl_sq(source);
                let nw = random_mask & nw_block;
                let ne = random_mask & ne_block;
                let se = random_mask & se_block;
                let sw = random_mask & sw_block;
                let mut test_board = Chessboard::new_blank();
                test_board.set_piece(Colors::WHITE, PieceType::BISHOP, source);
                test_board.set_piece(Colors::BLACK, PieceType::PAWN, random_mask);
                let legal_moves = test_board.generate_pseudolegal_bishop_moves(Colors::WHITE);
                for mv in legal_moves {
                    if mv.source != 0 {
                        assert_closest_bounds_ray(mv.source, nw);
                        assert_closest_bounds_ray(mv.source, ne);
                        assert_closest_bounds_ray(mv.source, sw);
                        assert_closest_bounds_ray(mv.source, se);
                    }
                }
            }
        }
    }

    #[test] // GPT generated this test
    fn test_queen_manual_example() {
        let mut board = Chessboard::new_blank();
        // Place a white queen on d4 (rank 3, file 3 → bit index 27)
        let queen_sq = 1u64 << (3 * 8 + 3);
        board.set_piece(Colors::WHITE, PieceType::QUEEN, queen_sq);

        // Put pawns to block: one on d6, one on f4, one on b2
        board.set_piece(Colors::BLACK, PieceType::PAWN,
            (1u64 << (5 * 8 + 3))   // d6
            | (1u64 << (3 * 8 + 5))   // f4
            | (1u64 << (1 * 8 + 1)),
        ); // b2

        let moves = board.generate_pseudolegal_queen_moves(Colors::WHITE);

        // Find queen's move set
        let queen_moves = moves.iter().find(|m| m.source == queen_sq).unwrap();

        // Assert that rays stop at the blockers
        assert!(queen_moves.dests & (1u64 << (5 * 8 + 3)) != 0); // includes d6
        assert!(queen_moves.dests & (1u64 << (6 * 8 + 3)) == 0); // not past d6
        assert!(queen_moves.dests & (1u64 << (3 * 8 + 5)) != 0); // includes f4
        assert!(queen_moves.dests & (1u64 << (3 * 8 + 6)) == 0); // not past f4
        assert!(queen_moves.dests & (1u64 << (1 * 8 + 1)) != 0); // includes b2
        assert!(queen_moves.dests & (1u64 << (0 * 8 + 0)) == 0); // not past b2
    }

    #[test]
    fn test_knight_blockers_move() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for i in 0usize..64 {
            let source = 1u64 << i;
            let mask = KNIGHT_MOVES[i];
            for _ in 0..10 {
                let random_mask = rng.random::<u64>() & mask;

                let mut test_board = Chessboard::new_blank();
                test_board.set_piece(Colors::WHITE, PieceType::KNIGHT, source);
                test_board.set_piece(Colors::BLACK, PieceType::KNIGHT, random_mask);

                let legal_moves = test_board.generate_pseudolegal_knight_moves(Colors::WHITE);

                for mv in legal_moves {
                    if mv.source == source {
                        let expected = KNIGHT_MOVES[i] & !test_board.get_piece(Colors::WHITE, PieceType::KNIGHT);
                        assert_eq!(mv.dests, expected);
                    }
                }
            }
        }
    }

    #[test] // GPT generated test
    fn test_king_blockers_move() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);
        for i in 0usize..64 {
            let source = 1u64 << i;
            let mask = KING_MOVES[i];
            for _ in 0..10 {
                let random_mask = rng.random::<u64>() & mask;

                let mut test_board = Chessboard::new_blank();
                test_board.set_piece(Colors::WHITE, PieceType::KING, source);
                test_board.set_piece(Colors::WHITE, PieceType::PAWN, random_mask); // own blockers

                let (pseudolegal_moves, count) = test_board.generate_pseudolegal_king_moves(Colors::WHITE);
                let pseudolegal_mask = pseudolegal_moves.iter().filter(|&x| !x.is_castling()).fold(0u64, |x, mv| x | mv.dest);

                // Expected: king moves minus own blockers
                let expected = KING_MOVES[i] & !random_mask;
                assert_eq!(pseudolegal_mask, expected);
            }
        }
    }

    #[test]
    fn test_white_pawn_simple_push() {
        let mut board = Chessboard::new_blank();

        // White pawn on e2
        let pawn = 1u64 << (1 * 8 + 3);
        board.set_piece(Colors::WHITE, PieceType::PAWN, pawn);

        let moves = board.generate_pseudolegal_white_pawn_moves();
        let mv = moves.iter().find(|&&m| m.source == pawn).unwrap();

        // Should include e3 and e4
        assert_ne!(mv.dests & (1u64 << (2 * 8 + 3)), 0);
        assert_ne!(mv.dests & (1u64 << (3 * 8 + 3)), 0);
        assert_eq!(moves.iter().filter(|x| x.source == pawn).count(), 1);
        assert_eq!(moves.iter().filter(|x| x.source != 0).count(), 1);
    }

    #[test]
    fn test_white_pawn_en_passant() {
        let mut board = Chessboard::new_blank();

        let white_pawn = 1u64 << (5 * 8 + 3);
        let black_pawn = 1u64 << (5 * 8 + 4);
        board.set_piece(Colors::WHITE, PieceType::PAWN, white_pawn);
        board.enpassant_takes = black_pawn;
        board.set_piece(Colors::BLACK, PieceType::PAWN, black_pawn);

        let moves = board.generate_pseudolegal_white_pawn_moves();
        let mv = moves.iter().find(|m| m.source == white_pawn).unwrap();
        assert_eq!(mv.dests, (black_pawn << 8) | (white_pawn << 8));
    }

    #[test]
    fn test_black_pawn_en_passant() {
        let mut board = Chessboard::new_blank();

        let black_pawn = 1u64 << (4 * 8 + 3); // pawn on e4
        let white_pawn = 1u64 << (4 * 8 + 4); // d3
        board.set_piece(Colors::BLACK, PieceType::PAWN, black_pawn);
        board.enpassant_takes = white_pawn;
        board.set_piece(Colors::WHITE, PieceType::PAWN, white_pawn);

        let moves = board.generate_pseudolegal_black_pawn_moves();
        let mv = moves.iter().find(|m| m.source == black_pawn).unwrap();
        assert_eq!(mv.dests, (white_pawn >> 8) | (black_pawn >> 8));
    }

    #[test]
    fn test_pawn_sequence() {
        let mut board = Chessboard::new_blank();

        let first_pawn = map_rank_and_file_to_sq(1, 2);
        let second_pawn = map_rank_and_file_to_sq(1, 3);
        let third_pawn = map_rank_and_file_to_sq(2, 4);
        let fourth_pawn = map_rank_and_file_to_sq(1, 5);
        let fifth_pawn = map_rank_and_file_to_sq(2, 6);

        let knight = map_rank_and_file_to_sq(2, 1); // this was a knight on the board i made on lichess
        let take2 = map_rank_and_file_to_sq(2, 2);
        let take3 = map_rank_and_file_to_sq(2, 3);
        let take4 = map_rank_and_file_to_sq(3, 6);
        board.set_piece(Colors::WHITE, PieceType::PAWN, first_pawn | second_pawn | third_pawn | fourth_pawn | fifth_pawn);
        board.set_piece(Colors::BLACK, PieceType::PAWN, take2 | take3 | take4);
        board.set_piece(Colors::BLACK, PieceType::KNIGHT, knight);
        let moves = board.generate_pseudolegal_white_pawn_moves();
        assert_eq!(moves.iter().filter(|x| x.source != 0).count(), 5);
        for mv in moves {
            if mv.source == first_pawn {
                assert_eq!(mv.dests, knight | take3);
            } else if mv.source == second_pawn {
                assert_eq!(mv.dests, take2);
            } else if mv.source == third_pawn {
                assert_eq!(mv.dests, third_pawn << 8);
            } else if mv.source == fourth_pawn {
                assert_eq!(mv.dests, (fourth_pawn << 8) | (fourth_pawn << 16));
            } else if mv.source == fifth_pawn {
                assert_eq!(mv.dests, 0);
            }
        }

        let take1 = knight;
        let mut nxt_board = Chessboard::new_blank();
        nxt_board.set_piece(Colors::BLACK, PieceType::PAWN, take1 | take2 | take3 | take4);
        let fourth_pawn = fourth_pawn << 16;
        nxt_board.set_piece(Colors::WHITE, PieceType::PAWN, first_pawn | second_pawn | third_pawn | fourth_pawn | fifth_pawn);
        let en_passant_location = map_rank_and_file_to_sq(3, 5);
        nxt_board.enpassant_takes = en_passant_location;
        let moves = nxt_board.generate_pseudolegal_black_pawn_moves();
        for mv in moves {
            if mv.source == take1 {
                assert_eq!(mv.dests, map_rank_and_file_to_sq(1, 1) | first_pawn);
            } else if mv.source == take2 {
                assert_eq!(mv.dests, second_pawn);
            } else if mv.source == take3 {
                assert_eq!(mv.dests, first_pawn);
            } else if mv.source == take4 {
                assert_eq!(mv.dests, en_passant_location >> 8);
            }
        }
    }

    #[test] // GPT generated TODO and it sucks, clean this up.
    fn test_king_pseudolegal_moves() {
        let mut board = Chessboard::new_blank();

        // White king on e1
        board.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 4));

        // Friendly blockers: d1 = queen, d2 = pawn, e2 = pawn
        board.set_piece(Colors::WHITE, PieceType::QUEEN, map_rank_and_file_to_sq(0, 3));
        board.set_piece(Colors::WHITE, PieceType::PAWN, map_rank_and_file_to_sq(1, 3) | map_rank_and_file_to_sq(1, 4));

        // Black rook on f1 (enemy piece, should be capturable)
        board.set_piece(Colors::BLACK, PieceType::ROOK, map_rank_and_file_to_sq(0, 5));

        let (king_moves, count) = board.generate_pseudolegal_king_moves(Colors::WHITE);

        // The king should not be able to move onto d1/e2/d2 (friendly pieces).
        assert_eq!(king_moves.iter().find(|&mv| mv.dest == map_rank_and_file_to_sq(0, 3)), None);
        assert_eq!(king_moves.iter().find(|&mv| mv.dest == map_rank_and_file_to_sq(1, 4)), None);
        assert_eq!(king_moves.iter().find(|&mv| mv.dest == map_rank_and_file_to_sq(1, 3)), None);

        // The king should be able to move to f1 (capture black rook).
        king_moves.iter().find(|&mv| mv.dest == map_rank_and_file_to_sq(0, 5)).unwrap();

        // The king should be able to move to f2 (empty square).
        king_moves.iter().find(|&mv| mv.dest == map_rank_and_file_to_sq(1, 5)).unwrap();
    }

    #[test] // GPT generated, manually corrected TODO and it sucks, clean this up.
    fn test_complicated_position() {
        let mut board = Chessboard::new_blank();

        // ---------------- White pieces ----------------
        board.set_piece(Colors::WHITE, PieceType::KING,
                        map_rank_and_file_to_sq(0, 4), // e1
        );
        board.set_piece(Colors::WHITE, PieceType::QUEEN,
                        map_rank_and_file_to_sq(0, 3), // d1
        );
        board.set_piece(Colors::WHITE, PieceType::ROOK,
                        map_rank_and_file_to_sq(0, 0)  // a1
                            | map_rank_and_file_to_sq(0, 7), // h1
        );
        board.set_piece(Colors::WHITE, PieceType::BISHOP,
                        map_rank_and_file_to_sq(3, 2), // c4
        );
        board.set_piece(Colors::WHITE, PieceType::KNIGHT,
                        map_rank_and_file_to_sq(2, 5), // f3
        );
        board.set_piece(Colors::WHITE, PieceType::PAWN,
                        map_rank_and_file_to_sq(1, 4)  // e2
                            | map_rank_and_file_to_sq(1, 3)  // d2
                            | map_rank_and_file_to_sq(1, 2), // c2
        );

        // ---------------- Black pieces ----------------
        board.set_piece(Colors::BLACK, PieceType::KING,
                        map_rank_and_file_to_sq(7, 4), // e8
        );
        board.set_piece(Colors::BLACK, PieceType::QUEEN,
                        map_rank_and_file_to_sq(7, 3), // d8
        );
        board.set_piece(Colors::BLACK, PieceType::ROOK,
                        map_rank_and_file_to_sq(7, 0)  // a8
                            | map_rank_and_file_to_sq(7, 7), // h8
        );
        board.set_piece(Colors::BLACK, PieceType::BISHOP,
                        map_rank_and_file_to_sq(4, 2), // c5
        );
        board.set_piece(Colors::BLACK, PieceType::KNIGHT,
                        map_rank_and_file_to_sq(5, 5), // f6
        );
        board.set_piece(Colors::BLACK, PieceType::PAWN,
                        map_rank_and_file_to_sq(6, 4)  // e7
                            | map_rank_and_file_to_sq(6, 3)  // d7
                            | map_rank_and_file_to_sq(6, 2), // c7
        );

        // Generate moves
        let white_rook_moves = board.generate_pseudolegal_rook_moves(Colors::WHITE);
        let white_bishop_moves = board.generate_pseudolegal_bishop_moves(Colors::WHITE);
        let white_knight_moves = board.generate_pseudolegal_knight_moves(Colors::WHITE);
        let white_queen_moves = board.generate_pseudolegal_queen_moves(Colors::WHITE);
        let (white_king_moves, count) = board.generate_pseudolegal_king_moves(Colors::WHITE);

        // White rook on a1: should move up to a7 and capture on a8
        let rook_a1 = map_rank_and_file_to_sq(0, 0);
        let rook_move = white_rook_moves
            .iter()
            .find(|m| m.source == rook_a1)
            .unwrap();
        assert!(rook_move.dests & map_rank_and_file_to_sq(6, 0) != 0); // a7 open
        assert!(rook_move.dests & map_rank_and_file_to_sq(7, 0) != 0); // a8 enemy rook capture allowed

        // White bishop on c4: should be blocked by black pawn on c7
        let bishop_c4 = map_rank_and_file_to_sq(3, 2);
        let bishop_move = white_bishop_moves
            .iter()
            .find(|m| m.source == bishop_c4)
            .unwrap();
        assert!(bishop_move.dests & map_rank_and_file_to_sq(6, 2) == 0); // c7 blocked
        assert!(bishop_move.dests & map_rank_and_file_to_sq(5, 4) != 0); // e6 diagonal open

        // White knight on f3: should be able to jump to g5
        let knight_f3 = map_rank_and_file_to_sq(2, 5);
        let knight_move = white_knight_moves
            .iter()
            .find(|m| m.source == knight_f3)
            .unwrap();
        assert!(knight_move.dests & map_rank_and_file_to_sq(4, 6) != 0); // g5 valid
        assert!(knight_move.dests & map_rank_and_file_to_sq(4, 4) != 0); // e5 valid

        // White king on e1: should only have f1 and f2
        let king_e1 = map_rank_and_file_to_sq(0, 4);
        assert_eq!(white_king_moves.iter().fold(0u64, |x, &mv| x | mv.dest), map_rank_and_file_to_sq(0, 5) | map_rank_and_file_to_sq(1, 5));

        // White queen on d1: should see d-file open until d7
        let queen_d1 = map_rank_and_file_to_sq(0, 3);
        assert_eq!(
            white_queen_moves[0].dests,
            map_rank_and_file_to_sq(0, 1) | map_rank_and_file_to_sq(0, 2)
        );
    }

    #[test]
    fn test_basic_play_move_takes() {
        let mut test_board = Chessboard::new_blank();

        // Place a black knight on c3 (rank 2, file 1)
        let black_knight = map_rank_and_file_to_sq(2, 1);
        test_board.set_piece(Colors::BLACK, PieceType::KNIGHT, black_knight);

        // Place a white pawn on d2 (rank 1, file 3)
        let pawn = map_rank_and_file_to_sq(1, 3);
        test_board.set_piece(Colors::WHITE, PieceType::PAWN, pawn);

        let knight_moves = test_board.generate_pseudolegal_knight_moves(Colors::BLACK);
        let takes_pawn = knight_moves.iter().find(|m| m.dests & pawn > 0).unwrap();
        let takes_pawn_mv = Move::new(takes_pawn.source, pawn & takes_pawn.dests, PieceType::KNIGHT, Colors::BLACK, 0);
        let nxt_board = test_board.play_move(&takes_pawn_mv);

        assert_eq!(
            nxt_board.get_piece(Colors::BLACK, PieceType::KNIGHT) & black_knight,
            0
        );
        assert_eq!(
            nxt_board.get_piece(Colors::BLACK, PieceType::KNIGHT),
            pawn
        );
        assert_eq!(
            nxt_board.get_piece(Colors::WHITE, PieceType::PAWN),
            0
        );
    }

    #[test]
    fn test_is_check() {
        let mut test_board = Chessboard::new_blank();
        test_board.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 0));
        test_board.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(0, 7));
        test_board.set_piece(Colors::BLACK, PieceType::QUEEN, map_rank_and_file_to_sq(7, 7));
        assert!(test_board.is_king_in_check(Colors::WHITE));

        let mut test_board2 = Chessboard::new_blank();
        test_board2.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(3, 3));
        test_board2.set_piece(Colors::WHITE, PieceType::ROOK, map_rank_and_file_to_sq(4, 4));
        test_board2.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 0));
        assert!(!test_board2.is_king_in_check(Colors::BLACK));
    }

    #[test]
    fn test_is_checkmate() {
        let mut test_board = Chessboard::new_blank();
        test_board.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 3));
        test_board.set_piece(Colors::BLACK, PieceType::QUEEN, map_rank_and_file_to_sq(1, 3));
        test_board.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(2, 3));
        assert!(!test_board.is_checkmate(Colors::BLACK));
        assert!(test_board.is_checkmate(Colors::WHITE));
    }
    
    #[test]
    fn test_is_stalemate() {
        let mut test_board = Chessboard::new_blank();
        let black_rooks = map_rank_and_file_to_sq(1, 1) | map_rank_and_file_to_sq(3, 5) | map_rank_and_file_to_sq(4, 4) | map_rank_and_file_to_sq(4, 2);
        test_board.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(2, 3));
        test_board.set_piece(Colors::BLACK, PieceType::ROOK, black_rooks);
        test_board.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(7, 7));
        assert!(!test_board.is_checkmate(Colors::WHITE));
        assert!(test_board.is_stalemate(Colors::WHITE))
    }

    #[test]
    fn test_enpassant_sequence() {  // TODO enpassant on edge of board
        let mut test_board = Chessboard::new_blank();
        let white_pawn = map_rank_and_file_to_sq(1, 2);
        let enp_to = white_pawn << 8;
        let black_pawn = map_rank_and_file_to_sq(3, 3);
        test_board.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 3));
        test_board.set_piece(Colors::WHITE, PieceType::PAWN, white_pawn);
        test_board.set_piece(Colors::BLACK, PieceType::PAWN, black_pawn);
        test_board.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(7, 3));

        let pawn_move = Move::new(test_board.get_piece(Colors::WHITE, PieceType::PAWN), white_pawn << 16, PieceType::PAWN, Colors::WHITE, 0);
        assert!(pawn_move.is_pawn_move_2sq());
        let nxt_board = test_board.play_move(&pawn_move);
        assert_eq!(nxt_board.get_piece(Colors::WHITE, PieceType::PAWN), pawn_move.dest);
        assert_eq!(nxt_board.enpassant_takes, white_pawn << 16);
        let black_moves = nxt_board.generate_pseudolegal_black_pawn_moves()[0];
        assert_eq!(black_moves.enpassant_takes, white_pawn << 16);
        assert_eq!(black_moves.dests, map_rank_and_file_to_sq(2, 3) | map_rank_and_file_to_sq(2, 2));
        assert_eq!(black_moves.color, Colors::BLACK);
        let (black_pawn_mvs, count) = black_moves.to_moves();
        let mv1 = black_pawn_mvs[0];
        let mv2 = black_pawn_mvs[1];
        assert!((mv1.is_enpassant() || mv2.is_enpassant()) && !(mv1.is_enpassant() && mv2.is_enpassant()));
        if mv1.is_enpassant() {
            let enp_board = nxt_board.play_move(&mv1);
            assert_eq!(mv1.enpassant_takes_sq(), white_pawn << 16);
            assert_eq!(enp_board.get_piece(Colors::WHITE, PieceType::PAWN), 0u64);
            assert_eq!(enp_board.get_piece(Colors::BLACK, PieceType::PAWN), enp_to);
        }
        else if mv2.is_enpassant() {
            let enp_board = nxt_board.play_move(&mv2);
            assert_eq!(mv2.enpassant_takes_sq(), white_pawn << 16);
            assert_eq!(enp_board.get_piece(Colors::WHITE, PieceType::PAWN), 0u64);
            assert_eq!(enp_board.get_piece(Colors::BLACK, PieceType::PAWN), enp_to);
        }
        else {
            unreachable!();
        }
    }

    #[test]
    fn test_promotion() {
        let mut test_board1 = Chessboard::new_blank();
        test_board1.set_piece(Colors::WHITE, PieceType::PAWN, map_rank_and_file_to_sq(6, 0));
        test_board1.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 7));
        test_board1.set_piece(Colors::BLACK, PieceType::KING, map_rank_and_file_to_sq(7, 7));
        let (mvs, count) = test_board1.generate_all_moves(Colors::WHITE);
        let pawn_moves: Vec<&Move> = mvs.iter().filter(|&mv| mv.piece_type == PieceType::PAWN).collect();
        assert_eq!(pawn_moves.len(), 4); // four promo types

        let promo_to_rook_mvs = *pawn_moves.iter().find(|&&mv| mv.promotion == PieceType::ROOK).unwrap();
        let rook_board1 = test_board1.play_move(promo_to_rook_mvs);
        assert!(rook_board1.is_king_in_check(Colors::BLACK));
        assert_eq!(rook_board1.get_piece(Colors::WHITE, PieceType::ROOK), map_rank_and_file_to_sq(7, 0));

        let promote_bishop = *pawn_moves.iter().find(|&&mv| mv.promotion == PieceType::BISHOP).unwrap();
        let bishop_board1 = test_board1.play_move(promote_bishop);
        assert!(!bishop_board1.is_king_in_check(Colors::BLACK));
        assert_eq!(bishop_board1.get_piece(Colors::WHITE, PieceType::BISHOP), map_rank_and_file_to_sq(7, 0));

        let promote_knight = *pawn_moves.iter().find(|&&mv| mv.promotion == PieceType::KNIGHT).unwrap();
        let knight_board1 = test_board1.play_move(promote_knight);
        assert!(!knight_board1.is_king_in_check(Colors::BLACK));
        assert_eq!(knight_board1.get_piece(Colors::WHITE, PieceType::KNIGHT), map_rank_and_file_to_sq(7, 0));

        let promote_queen = *pawn_moves.iter().find(|&&mv| mv.promotion == PieceType::QUEEN).unwrap();
        let queen_board1 = test_board1.play_move(promote_queen);
        assert!(queen_board1.is_king_in_check(Colors::BLACK));
        assert_eq!(queen_board1.get_piece(Colors::WHITE, PieceType::QUEEN), map_rank_and_file_to_sq(7, 0));
    }

    #[test]
    fn test_kingside_castling() {
        let mut testboard = Chessboard::new_blank();
        testboard.kingside_castling_rights = [true, true];
        testboard.queenside_castling_rights = [true, true];
        testboard.set_piece(Colors::WHITE, PieceType::KING, map_rank_and_file_to_sq(0, 4));
        testboard.set_piece(Colors::WHITE, PieceType::ROOK, map_rank_and_file_to_sq(0, 0) | map_rank_and_file_to_sq(0, 7));
        let (mvs, count) = testboard.generate_all_moves(Colors::WHITE);
        let queenside_castle = mvs.iter().find(|&mv| mv.is_queenside_castle()).unwrap();
        let kingside_castle = mvs.iter().find(|&mv| mv.is_kingside_castle()).unwrap();
        let castled = testboard.play_move(kingside_castle);
        assert_eq!(castled.get_piece(Colors::WHITE, PieceType::KING), map_rank_and_file_to_sq(0, 6));
        assert_eq!(castled.get_piece(Colors::WHITE, PieceType::ROOK), map_rank_and_file_to_sq(0, 5));
    }
}
