use std::collections::HashSet;

use futures;
use tokio;
use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use crate::chessboard::Chessboard;
use crate::chessboard::Move;
use crate::chessboard::Colors;
use crate::datastructures::*;
use crate::inference_primitives::*;


pub enum TerminalState {
    WIN,
    LOSS,
    DRAW,
}


// Core idea is same as usual MCTS but replacing rollout with NN eval
pub struct ChessTree {
    expanded_arena: Arena<ExpandedPositionNode>,  // re-rooting drops children,
    deferred_arena: Arena<DeferredPositionNode>, // TODO subtree block arenas
    history: Vec<Chessboard>,
    total_move_count: u32,
    rt: tokio::runtime::Runtime,
}


struct ExpandedPositionNode {
    index: usize,
    parent: usize,  // root when index == parent
    action: Option<Move>, // action taken to reach this node from the parent
    chessboard: Chessboard,
    value_sum: f64,
    visit_count: u32,
    probability: f64,  // assigned on node expansion
    children: Array<usize, 256>,
    deferred_children: HashSet<usize>,
    color_to_play: Colors,
    priors: PositionPrior, // TODO deferred children
}

impl ExpandedPositionNode {
    /// The expectation is that this function is called with the popped deferred node
    /// Inserts the node into the tree and returns a reference
    fn from_deferred_node_and_push(tree: &mut ChessTree, deferred_node: DeferredPositionNode) -> &ExpandedPositionNode {
        let inference_result = tree.block_on(deferred_node.inference_result).unwrap();
        let me = ExpandedPositionNode {
            index: 0, // set correctly below
            parent: deferred_node.parent,
            action: deferred_node.action,
            chessboard: deferred_node.chessboard,
            value_sum: inference_result.get_value(),
            visit_count: 1,
            probability: deferred_node.probability,
            children: Array::new(),
            deferred_children: HashSet::new(),
            color_to_play: deferred_node.color_to_play,
            priors: inference_result.get_priors(),
        };
        me.get_parent_mut(tree).deferred_children.remove(&deferred_node.index);
        let ind = tree.expanded_arena.push(me);
        tree.expanded_arena.get_mut(ind).unwrap().index = ind;
        tree.expanded_arena.get(ind).unwrap()
    }

    fn is_root(&self) -> bool {
        self.index == self.parent
    }

    fn mean_action_value_from_s(&self) -> f64 {
        self.value_sum / (self.visit_count as f64)
    }

    fn ucb(&self) -> f64 { // U(s, a), s is the parent, a is self
        self.mean_action_value_from_s() + self.probability * ((self.visit_count as f64).sqrt() / (1f64 + self.visit_count as f64))  // TODO constant for puct?
    }

    fn is_visited(&self) -> bool {
        self.visit_count > 0
    }

    fn get_value(&self) -> f64 {
        self.value_sum
    }

    fn get_probability(&self, dirichlet_param: f64) -> f64 {
        self.probability // TODO dirichlet noise
    }

    fn increase_value(&mut self, value: f64) {
        self.visit_count += 1;
        self.value_sum += value;
    }

    fn get_parent_mut(&self, tree: &ChessTree) -> &mut ExpandedPositionNode {
        tree.expanded_arena.get_mut(self.parent).unwrap()
    }

}

struct DeferredPositionNode {
    index: usize,
    parent: usize,
    action: Option<Move>,
    chessboard: Chessboard,
    inference_result: tokio::task::JoinHandle<PositionInferenceResult>,
    probability: f64,  // from parent
    color_to_play: Colors,
}

impl DeferredPositionNode {
    fn new_and_push(tree: &mut ChessTree, parent: &ExpandedPositionNode, mv: Move) -> usize {
        let board = parent.chessboard.play_move(&mv);
        let me = DeferredPositionNode {
            index: 0,  // reset correctly below
            parent: parent.index,
            probability: parent.priors.to_prob(&mv),
            action: Some(mv),
            chessboard: board.clone(),
            inference_result: tree.spawn(async { PositionInferenceResult::from_chessboard(board).await } ),
            color_to_play: Colors::opposite_color(parent.color_to_play),
        };
        let ind = tree.deferred_arena.push(me);
        tree.deferred_arena.get_mut(ind).unwrap().index = ind;
        ind
    }

    fn is_root(&self) -> bool {
        self.index == self.parent
    }

    pub fn new_game_root(tree: &mut ChessTree) -> usize {
        let board = Chessboard::new();
        let me = DeferredPositionNode {
            index: 0,
            parent: 0,
            action: None,
            chessboard: board,
            inference_result: tree.spawn(async { PositionInferenceResult::from_chessboard(board.clone()).await } ),
            probability: 1f64,
            color_to_play: Colors::WHITE,
        }; // TODO will spawn every inference task so it starts running and await on rollout ()
        tree.deferred_arena.push(me)
    }

    pub fn to_tensor(&self, tree: &ChessTree) -> ChessInferenceTensor {
        let meta = self.chessboard.to_mv_metadata_tensor(self.color_to_play, tree.total_move_count);
        let mut mvs: Array<MoveTensor, 8> = Array::new();
        mvs.push(self.chessboard.to_mv_tensor(self.color_to_play, tree.get_repetition_count(&self.chessboard)));
        let mut history_count: usize = 7;
        let mut cur = tree.expanded_arena.get(self.parent).unwrap();
        while history_count != 0 && !cur.is_root() {
            mvs.push(cur.to_mv_tensor());
            cur = &cur.get_parent();
            history_count -= 1;
        }
        for i in 0..std::cmp::min(history_count, tree.history.len()) {
            mvs.push(tree.history[tree.history.len() - 1 - i].to_mv_tensor(self.color_to_play, tree.get_repetition_count(&self.chessboard)));
            history_count -= 1;
        }
        for _ in 0..history_count {
            mvs.push(MoveTensor::new_zeros());
        }
        ChessInferenceTensor::new(mvs.as_raw(), meta)
    }

    pub fn get_terminal_position(&self, tree: &ChessTree) -> Option<TerminalState> {
        if self.chessboard.king_is_checkmated(self.color_to_play) {
            Some(TerminalState::LOSS)
        } else if self.chessboard.king_is_checkmated(Colors::opposite_color(self.color_to_play)) {
            Some(TerminalState::WIN)
        } else if self.chessboard.is_draw_by_insufficient_material_or_50_move_rule() || tree.get_repetition_count(&self.chessboard) == 2 {
            Some(TerminalState::DRAW)
        } else {
            None
        }
    }
}

impl ChessTree {
    // concurrency utils
    fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(async move {
            fut.await;
        })
    }

    fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.rt.block_on(future)
    }

    pub fn new_with_inference() -> ChessTree {
        let mut me = ChessTree {
            expanded_arena: Arena::new(),
            deferred_arena: Arena::new(),
            history: Vec::new(),
            total_move_count: 0,
            rt: tokio::runtime::Runtime::new().unwrap(),
        };

        let root_ind = DeferredPositionNode::new_game_root(&mut me);
        let root = me.deferred_arena.pop(root_ind); // reuses inference req code path
        ExpandedPositionNode::from_deferred_node_and_push(&mut me, root);

        me
    }

    pub fn reroot(&mut self, temperature: f64) {
        // TODO also have a reroot method which maintains the explored subtree
        // TODO send node board, value, and probabilities to replay buffer
        let root = self.get_root();
        let probs = self.get_action_probabilities(temperature);
        let next_root = if temperature <= 1e-9 {
            root.children.iter().map(|&i| self.expanded_arena.get(i).unwrap()).max_by_key(|&x| x.visit_count).unwrap()
        } else {
            let mut rng = rng();
            let dist = WeightedIndex::new(&probs).unwrap();
            let child_index = dist.sample(&mut rng);
            let global_index = root.children[child_index];
            self.expanded_arena.get(global_index).unwrap()
        };
        self.history.push(root.chessboard);
        self.deferred_arena = Arena::new();
        self.expanded_arena = Arena::new();
        self.expanded_arena.push(*next_root.clone());
    }

    fn get_repetition_count(&self, board: &Chessboard) -> u32 {
        self.history.iter().filter(|&b| b == board).count() as u32 // TODO replace with zobrist eq
    }

    fn get_action_probabilities(&self, temperature: f64) -> Vec<f64> {
        let total = self.get_root().children.iter().fold(0f64, |acc, &i| acc + (self.expanded_arena.get(i).unwrap().visit_count as f64).powf(1f64 / temperature));
        self.get_root().children.iter().map(|&i| (self.expanded_arena.get(i).unwrap().visit_count as f64).powf(1f64 / temperature) / total).collect()
    }

    fn select_child(&self, cur: &ExpandedPositionNode) -> Result<&ExpandedPositionNode, usize> {
        let best_expanded_child = cur.children.iter().map(|&i| self.expanded_arena.get(i).unwrap()).reduce(|x, y| {
            if x.ucb() > y.ucb() { x } else { y }
        });
        match best_expanded_child {
            Some(child) => Ok(child),
            None => {
                Err(*cur.deferred_children.iter().last().unwrap())
            }
        }
    }

    pub fn get_root(&self) -> &ExpandedPositionNode {
        self.expanded_arena.get(0).unwrap()
    }

    fn select_and_rollout(&mut self) {
        let mut cur = self.get_root();
        loop {
            let preferred_child = self.select_child(cur);
            match preferred_child {
                Ok(child) => cur = child,
                Err(child_index) => {
                    self.rollout(child_index);
                    return;
                }
            }
        }
    }

    pub fn simulate(&mut self, num_simulations: u32) {
        for i in 0..num_simulations {
            self.select_and_rollout()
        }
    }

    // TODO try kalmogorov network (joe weber sent this to me)
    // TODO set illegal logits to -inf so they don't get gradient signal

    fn expand_node(&mut self, node: &mut ExpandedPositionNode) {
        if !node.is_visited() {
            let (nxt_boards, count) = node.chessboard.generate_next_legal_boards(node.color_to_play);
            for i in 0..count {
                let (board, mv) = nxt_boards[i]; // TODO board & mv should probably not be Copy and just use clone where needed
                let child_node = DeferredPositionNode::new_and_push(self, node, mv);
                node.deferred_children.insert(child_node);
            }

        }
    }

    /// Applies the deferred inference result & invalidates the deferred node index
    fn rollout(&mut self, deferred_node_index: usize) {
        let node = self.deferred_arena.pop(deferred_node_index);
        let mut cur_node = ExpandedPositionNode::from_deferred_node_and_push(self, node);
        let value = cur_node.value_sum;
        while !cur_node.is_root() {
            let mut parent = cur_node.get_parent_mut(self);
            parent.increase_value(value);
            cur_node = parent;
        }
    }
}
