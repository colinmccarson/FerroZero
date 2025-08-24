pub fn print_u64_as_8x8_bit_string(n: u64) {
    let binary_string = format!("{:064b}", n);

    for (i, c) in binary_string.chars().enumerate() {
        print!("{}", c);
        if (i + 1) % 8 == 0 { 
            print!("\n");
        }
        if (i + 1) % 64 == 0 {
            println!();
        }
    }
}


pub fn get_rank_index(mv: u64) -> Option<u64> {
    for i in 0u64..8 {
        if mv & (0xFF << (i << 3)) > 0 {
            return Some(i);
        }
    }
    return None;
}

pub fn get_file_index(mv: u64) -> Option<u64> {
    for i in 0u64..64 {
        if mv & (1 << i) > 0 {
            return Some(7 - (i % 8));
        }
    }
    return None;
}


pub fn collect_coordinates(mv_set: u64) -> Vec<(u64, u64)> {
    let mut v: Vec<(u64, u64)> = Vec::new();
    for i in 0u64..64 {
        let mv: u64 = 1 << i;
        if (mv & mv_set) > 0 {
            v.push((get_rank_index(mv).unwrap(), get_file_index(mv).unwrap()));
        }
    }
    v
}
