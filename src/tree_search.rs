use futures;
use tokio;

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
    arena: Vec<Vec<ChessNode>>,  // a child of the root is always index [i][0]
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
    color_to_play: Colors,
}


impl ChessNode {
    pub fn make_root() {
        // TODO
    }
    pub fn new(root_index: usize, index: usize, parent: usize, action: Move, chessboard: Chessboard, color_to_play: Colors) -> ChessNode {
        ChessNode {
            root_index,
            index,
            parent: Some(parent),
            action,
            chessboard,
            value_sum: 0f64,
            visit_count: 0u32,
            children: Vec::new(),
            probability: 0f64,
            color_to_play
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
        TerminalState::DRAW
    }

    pub fn get_probability(&self, dirichlet_param: f64) -> f64 {
        self.probability // TODO dirichlet noise
    }

}


impl ChessTree {
    pub fn create_children_for_node(&mut self, root_children_index: usize, node_index: usize) {
        let node = &self.arena[root_children_index][node_index];
        let next_color = Colors::opposite_color(node.color_to_play);
        let (result, count) = node.chessboard.generate_next_legal_boards(node.color_to_play);
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
                    next_color
                )
            )
        }
    }
    
    pub fn reroot(&self, root_child_index: usize) {
        // TODO
    }

    pub fn get_action_probabilities(&self, temperature: f64) -> Vec<f64> {
        let total = self.root.children.iter().fold(0f64, |acc, &i| acc + (self.arena[i][0].visit_count as f64).powf(1f64 / temperature));
        self.root.children.iter().map(|&i| (self.arena[i][0].visit_count as f64).powf(1f64 / temperature) / total).collect()
    }
    
    pub fn select_child(&self, cur: &ChessNode) -> &ChessNode {
        cur.children.iter().map(|&i| &self.arena[cur.root_index][i]).reduce(|x, y| if x.ucb() > y.ucb() { x } else { y }).unwrap()
    }

    pub fn select(&self) -> &ChessNode {
        let mut cur = &self.root;
        while cur.is_visited() {
            cur = self.select_child(cur);
        }
        cur
    }

    async fn get_position_probability_and_value(position: Chessboard) -> (f64, f64) {
        // TODO
        (0f64, 0f64)
    }

    // TODO NN takes in a board and spits out 64^2 probabilities
    // TODO try kalmogorov network (joe weber sent this to me)
    // TODO set illegal logits to -inf so they don't get gradient signal

    // TODO this should support batch processing
    pub fn expand_node(&mut self, node: &mut ChessNode) {
        let mut handles = Vec::new();
        if !node.is_visited() {
            let (nxt_boards, count) = node.chessboard.generate_next_legal_boards(node.color_to_play);
            for i in 0..count {
                let (board, mv) = nxt_boards[i]; // TODO board & mv should probably not be Copy and just use clone where needed
                let child_node = ChessNode::new(
                    node.root_index,
                    node.children.len(),
                    node.index,
                    mv,
                    board.clone(),
                    Colors::opposite_color(node.color_to_play)
                );
                // TODO this will be a coroutine for each state and expand_node only returns when each coroutine comes back
                let handle = tokio::task::spawn(async move {
                    ChessTree::get_position_probability_and_value(board).await
                });
                handles.push(handle);
                node.children.push(child_node.index);
                self.arena[node.root_index].push(child_node);
            }
        }
        let results = tokio::runtime::Handle::current().block_on(async {
            futures::future::join_all(handles).await
        });
        // TODO backpropagation
    }
}
