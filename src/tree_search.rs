use crate::chessboard::Chessboard;
use crate::chessboard::Move;
use crate::chessboard::Colors;


pub enum TerminalState {
    WIN,
    LOSS,
    DRAW,
}


// Core idea is same as usual MCTS but replacing rollout with NN eval
pub struct ChessTree {
    root: ChessNode,
    arena: Vec<Vec<ChessNode>>,  // will sometimes need to be GC'd after re-rooting, TODO
}


pub struct ChessNode {
    root_index: usize,
    index: usize,
    parent: Option<usize>,
    action: Move,
    chessboard: Chessboard,
    value_sum: f64,
    visit_count: u32,
    probability: f64,  // assigned on node expansion
    children: Vec<usize>,
}


impl ChessNode {
    pub fn make_root() {
        // TODO
    }
    pub fn new(root_index: usize, index: usize, parent: usize, action: Move, chessboard: Chessboard) -> ChessNode {
        ChessNode {
            root_index: root_index,
            index: index,
            parent: Some(parent),
            action: action,
            chessboard: chessboard,
            value_sum: 0f64,
            visit_count: 0u32,
            children: Vec::new(),
            probability: 0f64,
        }
    }
    fn mean_action_value_from_s(&self) -> f64 {
        self.value_sum / (self.visit_count as f64)
    }

    pub fn ucb(&self) -> f64 { // U(s, a), s is the parent, a is self
        if self.visit_count == 0 {
            return f64::INFINITY
        }
        self.mean_action_value_from_s() + self.probability * ((self.visit_count as f64).sqrt() / (1f64 + self.visit_count as f64))  // TODO constant for puct?
    }

    pub fn is_visited(&self) -> bool {
        self.visit_count > 0
    }

    pub fn get_value(&self) -> f64 { // TODO NN
        0f64
    }

    pub fn terminal(&self) -> TerminalState {
        
    }

}


impl ChessTree {
    pub fn create_children_for_node(&mut self, root_children_index: usize, node_index: usize, color: Colors) {
        let node = &self.arena[root_children_index][node_index];
        let (result, count) = node.chessboard.generate_next_legal_boards(color);
        for i in 0..count {
            let (board, mv) = result[i];
            let ind = self.arena[root_children_index].len();
            self.arena[root_children_index].push(
                ChessNode::new(
                    root_children_index,
                    ind,
                    node_index,
                    mv,
                    board,
                )
            )
        }
    }
    
    pub fn reroot(&self, root_child_index: usize) {
        // TODO
    }
    pub fn get_action_probabilities(&self, temperature: f64) -> Vec<f64> {
        let total = self.root.children.iter().fold(0f64, |acc, &i| acc + (self.arena[i].visit_count as f64).powf(1f64 / temperature));
        self.root.children.iter().map(|&i| (self.arena[i].visit_count as f64).powf(1f64 / temperature) / total).collect()
    }
    
    pub fn select_child(&self, cur: &ChessNode) -> &ChessNode {
        cur.children.iter().map(|&i| &self.arena[i]).reduce(|x, y| if x.ucb() > y.ucb() { x } else { y }).unwrap()
    }

    pub fn select(&self) -> &ChessNode {
        let mut cur = &self.root;
        while cur.is_visited() {
            cur = self.select_child(cur);
        }
        cur
    }
    

    // TODO backpropagation
}
