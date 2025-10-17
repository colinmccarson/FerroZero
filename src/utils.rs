#[inline(always)]
pub fn bool_to_mask(b: bool) -> u64 {
    -(b as i64) as u64
}

#[inline(always)]
pub fn branchless_select(b: bool, if_false: u64, if_true: u64) -> u64 {  // TODO use this everywhere possible
    let m = bool_to_mask(b);
    (if_false & !m) | (if_true & m)
}
