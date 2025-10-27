use std::rc::Rc;

use futures;
use tch::Tensor;
use tokio;

use crate::chessboard::Chessboard;
use crate::chessboard::Move;
use crate::chessboard::Colors;
use crate::datastructures::*;
use crate::inference_primitives::*;


type PositionIndexArray = Array<usize, 256>;


pub enum TerminalState {
    WIN,
    LOSS,
    DRAW,
}


// Core idea is same as usual MCTS but replacing rollout with NN eval
pub struct ChessTree {
    root: PositionNode,
    arena: Vec<PositionNode>,  // re-rooting drops children
    history: Rc<RingBuffer<Chessboard, 7>>,
    total_move_count: u32,
}


pub struct ExpandedPositionNode {
    tree: std::rc::Rc<ChessTree>,
    index: usize,
    parent: usize,  // root when index == parent
    action: Move,
    chessboard: Chessboard,
    value_sum: f64,
    visit_count: u32,
    probability: f64,  // assigned on node expansion
    children: PositionIndexArray,
    color_to_play: Colors,
    deferred_value: Option<f64>,
    deferred_probability: Option<f64>,
}

impl ExpandedPositionNode {
    pub fn make_root() {
        // TODO
    }
    pub fn new(root_index: usize, index: usize, parent: usize, action: Move, chessboard: Chessboard, color_to_play: Colors) -> ExpandedPositionNode {
        ExpandedPositionNode {
            index,
            action,
            chessboard,
            value_sum: None,
            visit_count: 0u32,
            children: Vec::new(),
            probability: None,
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

    pub fn defer_value_and_prior() {
        todo!()
    }

    pub fn apply_deferred_value_and_prior(&mut self, ) {
        todo!()
    }

}

pub struct DeferredPositionNode {
    tree: Rc<ChessTree>,
    index: usize,
    parent: usize,
    action: Move,
    chessboard: Chessboard,
    deferred_value: f64,
    deferred_priors: PositionPrior,
    probability: f64,  // from parent
    color_to_play: Colors,
}

impl DeferredPositionNode {

    pub fn to_tensor(&self) -> ChessInferenceTensor {
        let meta = self.chessboard.to_mv_metadata_tensor(self.color_to_play, self.tree.total_move_count);
        let mut mvs: Array<MoveTensor, 8> = Array::new();
        mvs.push(self.chessboard.to_mv_tensor(self.color_to_play, self.tree.get_repetition_count(&self.chessboard)));
        let mut history_count: usize = 7;
        let mut cur = &self.tree.arena.get(self.parent).unwrap();
        while history_count != 0 && !cur.is_root() {
            mvs.push(cur.to_mv_tensor());
            cur = &cur.get_parent();
            history_count -= 1;
        }
        for i in 0..history_count {
            mvs.push(self.tree.history[i].to_mv_tensor(self.color_to_play, self.tree.get_repetition_count(&self.chessboard)));
        }
        ChessInferenceTensor::new(mvs.as_raw(), meta)
    }
}

pub enum PositionNode {
    Expanded(ExpandedPositionNode),
    Deferred(DeferredPositionNode),
}

impl PositionNode { // This is just a dispatch layer
    pub fn get_parent(&self) -> &PositionNode {
        match self {
            PositionNode::Deferred(node)  => {
                node.tree.arena.get(node.parent).unwrap()
            },
            PositionNode::Expanded(node) => {
                node.tree.arena.get(node.parent).unwrap()
            }
        }
    }

    pub fn is_root(&self) -> bool {
        match self {
            PositionNode::Expanded(node) => {
                node.parent == node.index
            }
            PositionNode::Deferred(node) => {
                node.parent == node.index
            }
        }
    }

    pub fn to_mv_tensor(&self) -> MoveTensor { // TODO if this is slow enough, maybe LRU cache
        match self {
            PositionNode::Deferred(node) => {
                node.chessboard.to_mv_tensor(node.color_to_play, node.tree.get_repetition_count(&node.chessboard))
            }
            PositionNode::Expanded(node) => {
                node.chessboard.to_mv_tensor(node.color_to_play, node.tree.get_repetition_count(&node.chessboard))
            }
        }
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
            let child = ExpandedPositionNode::new(
                root_children_index,
                ind,
                node_index,
                mv,
                board,
                next_color
            );
            let arena_len = self.arena[root_children_index].len();
            self.arena[root_children_index][node_index].children.push(arena_len);  // reborrow as mut
            self.arena[root_children_index].push(child)
        }
    }
    
    pub fn reroot(&self, root_child_index: usize) {
        todo!()
    }

    pub fn get_repetition_count(&self, &Chessboard) -> u32 {
        todo!()
    }

    pub fn get_action_probabilities(&self, temperature: f64) -> Vec<f64> {
        let total = self.root.children.iter().fold(0f64, |acc, &i| acc + (self.arena[i][0].visit_count as f64).powf(1f64 / temperature));
        self.root.children.iter().map(|&i| (self.arena[i][0].visit_count as f64).powf(1f64 / temperature) / total).collect()
    }
    
    pub fn select_child(&self, cur: &ExpandedPositionNode) -> &ExpandedPositionNode {
        cur.children.iter().map(|&i| &self.arena[cur.root_index][i]).reduce(|x, y| if x.ucb() > y.ucb() { x } else { y }).unwrap()
    }

    pub fn select(&self) -> &ExpandedPositionNode {
        let mut cur = &self.root;
        while cur.is_visited() {
            cur = self.select_child(cur);
        }
        cur
    }

    async fn get_position_value_and_priors(position_node: &DeferredPositionNode) -> (f64, Vec<f64>) { // TODO second element here should have return type from inference results processor, and will need to be renormalized
        // returns f(s), pi(a | s)
        // TODO
    } // TODO make the above return type stack allocated, so need to implement MoveList

    // TODO NN takes in a board and spits out 64^2 probabilities
    // TODO try kalmogorov network (joe weber sent this to me)
    // TODO set illegal logits to -inf so they don't get gradient signal

    // TODO this should support batch processing
    pub fn expand_node(&mut self, node: &mut ExpandedPositionNode) {
        if !node.is_visited() {
            let mut handles = Vec::new();
            let (nxt_boards, count) = node.chessboard.generate_next_legal_boards(node.color_to_play);
            for i in 0..count {
                let (board, mv) = nxt_boards[i]; // TODO board & mv should probably not be Copy and just use clone where needed
                let child_node = ExpandedPositionNode::new(
                    node.root_index,
                    node.children.len(),
                    node.index,
                    mv,
                    board.clone(),
                    Colors::opposite_color(node.color_to_play)
                );
                // TODO cache & defer application of values from NN, 'pre-fetching'
                let handle = tokio::spawn(async move {
                    ChessTree::get_position_value_and_priors(&board).await
                });
                handles.push(handle);
                node.children.push(child_node.index);
                self.arena[node.root_index].push(child_node);
            }
            let rt = tokio::runtime::Runtime::new().unwrap();

            let results = rt.block_on(async {
                futures::future::join_all(handles).await
            });
            // TODO these are index aligned, so can just write the result to node.children[i]

        }
    }

    pub fn rollout(&mut self, node: &mut ExpandedPositionNode) { // TODO apply pre-fetched result; only the root will not be pre-fetched.
        let (value, priors) = ChessTree::get_position_value_and_priors(&node.chessboard);
        self.create_children_for_node(node.root_index, node.index);
        // TODO backpropagation
    }
}
