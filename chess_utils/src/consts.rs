pub const RANK_8: u64 = 0xFF00_0000_0000_0000;
pub const RANK_7: u64 = 0x00FF_0000_0000_0000;
pub const RANK_2: u64 = 0x0000_0000_0000_FF00;
pub const RANK_1: u64 = 0x0000_0000_0000_00FF;
pub const FILE_A: u64 = 0x8080_8080_8080_8080;
pub const FILE_H: u64 = 0x0101_0101_0101_0101;
pub const BOARD_EDGES: u64 = RANK_1 | RANK_8 | FILE_A | FILE_H;
pub const BOARD_CORNERS: u64 = (RANK_1 & FILE_A) | (RANK_1 & FILE_H) | (RANK_8 & FILE_A) | (RANK_8 & FILE_H);

#[repr(usize)]
#[derive(Clone)]
pub enum DIRECTIONS {
    N = 0,
    NE = 1,
    E = 2,
    SE = 3,
    S = 4, 
    SW = 5,
    W = 6,
    NW = 7,
}

impl DIRECTIONS {
    pub fn from_usize(n: usize) -> DIRECTIONS {
        match n {
            0 => DIRECTIONS::N,
            1 => DIRECTIONS::NE,
            2 => DIRECTIONS::E,
            3 => DIRECTIONS::SE,
            4 => DIRECTIONS::S,
            5 => DIRECTIONS::SW,
            6 => DIRECTIONS::W,
            7 => DIRECTIONS::NW,
            _ => panic!("Directions are 0-7 only"),
        }
    }
}
