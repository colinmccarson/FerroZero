use alphazero::tree_search::*;

fn main() {
    let mut chessgame = ChessTree::new_with_inference(0.5);
    chessgame.simulate(100);
}
