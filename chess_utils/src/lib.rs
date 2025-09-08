pub mod utils;
pub mod shifts;
pub mod consts;

#[cfg(test)]
mod tests {
    use crate::utils::*;

    fn bb(rank: usize, file: usize) -> u64 {
        1u64 << (rank * 8 + file)
    }

    #[test]
    fn rook_attack_empty_board_center() {
        let sq = bb(3, 3); // d4
        let occ = 0;
        let attack = brute_force_rook_attack(sq, occ);

        // Expect all squares on rank/file except the source itself
        let mut expected = 0u64;
        // same file
        for r in 0..8 {
            if r != 3 {
                expected |= bb(r, 3);
            }
        }
        // same rank
        for f in 0..8 {
            if f != 3 {
                expected |= bb(3, f);
            }
        }
        assert_eq!(attack, expected);
    }

    #[test]
    fn rook_attack_with_blocker() {
        let sq = bb(0, 0); // a1
        let occ = bb(0, 3); // blocker at d1
        let attack = brute_force_rook_attack(sq, occ);

        // Should include a2..a8 vertically, and b1..d1 horizontally
        let mut expected = 0u64;
        for r in 1..8 {
            expected |= bb(r, 0); // file a
        }
        for f in 1..=3 {
            expected |= bb(0, f); // rank 1 up to d1
        }
        assert_eq!(attack, expected);
    }

    #[test]
    fn bishop_attack_empty_board_center() {
        let sq = bb(3, 3); // d4
        let occ = 0;
        let attack = brute_force_bishop_attack(sq, occ);

        let mut expected = 0u64;
        // NW
        let mut r = 4;
        let mut f = 2;
        while r < 8 && f < 8 {
            expected |= bb(r, f);
            r += 1;
            if f == 0 { break; }
            f -= 1;
        }
        // NE
        let mut r = 4;
        let mut f = 4;
        while r < 8 && f < 8 {
            expected |= bb(r, f);
            r += 1;
            f += 1;
        }
        // SW
        let mut r = 2i32;
        let mut f = 2i32;
        while r >= 0 && f >= 0 {
            expected |= bb(r as usize, f as usize);
            r -= 1;
            f -= 1;
        }
        // SE
        let mut r = 2i32;
        let mut f = 4i32;
        while r >= 0 && f < 8 {
            expected |= bb(r as usize, f as usize);
            r -= 1;
            f += 1;
        }

        assert_eq!(attack, expected);
    }

    #[test]
    fn bishop_attack_with_blocker() {
        let sq = bb(0, 0);
        let occ = bb(2, 2); 
        let attack = brute_force_bishop_attack(sq, occ);

        let expected = bb(1, 1) | bb(2, 2);
        assert_eq!(attack, expected);
    }
}
