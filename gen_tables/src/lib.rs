include!(concat!(env!("OUT_DIR"), "/generated_chess_tables.rs"));


#[test]
fn it_works() {
    assert_eq!(HELLO, "chess tables ready");
}

