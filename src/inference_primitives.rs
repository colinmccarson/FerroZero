use bytemuck;
use std::borrow::Borrow;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeStruct;
use tch;
use tch::IndexOp;
use chess_utils::utils::get_direction_to_target;
use chess_utils::utils::get_distance_in_direction_not_zero;
use chess_utils::utils::map_rank_and_file_to_sq;
use chess_utils::utils::get_file_index;
use chess_utils::utils::get_rank_index;

use crate::chessboard::Chessboard;
use crate::chessboard::Colors;
use crate::chessboard::Move;
use crate::datastructures::Array;


#[inline]
fn map_8x8_plane_idx_to_sq(i: i32, j: i32) -> Option<u64> {
    if i < 0 || i > 7 || j < 0 || j > 7 {
        return None;
    }
    let rank = (7 - i) as u64;
    let file = j as u64;
    Some(map_rank_and_file_to_sq(rank, file))
}

#[inline]
fn map_sq_to_8x8_plane_idx(sq: u64) -> (i32, i32) {
    let rank = get_rank_index(sq).unwrap() as i32;
    let file = get_file_index(sq).unwrap() as i32;
    (7 - rank, file)
}


/// Indexing scheme in the first 56 is (direction, num squares)
#[derive(Debug)]
pub struct PositionPrior(tch::Tensor);

impl PositionPrior {
    const EXPECTED_SHAPE: [i64; 3] = [8, 8, 73];

    pub fn new() -> Self {
        Self(tch::Tensor::zeros(&Self::EXPECTED_SHAPE, (tch::Kind::Half, tch::Device::Cpu)))
    }
    
    pub fn from_tensor(tens: tch::Tensor) -> Self {
        Self(tens)
    }

    fn tens_idx_from_mv(mv: &Move) -> (i32, i32, i32) {
        let (i, j) = map_sq_to_8x8_plane_idx(mv.get_src());
        let dir = get_direction_to_target(mv.get_src(), mv.get_dest()).unwrap();
        let distance = get_distance_in_direction_not_zero(mv.get_src(), mv.get_dest(), dir.clone());
        (i, j, 8 * (dir as i32) + distance)
    }

    pub fn to_probs(&self, legal_mvs: &Array<Move, 256>) -> Array<f64, 256> {
        let mut probs: Array<f64, 256> = Array::new();
        for mv in legal_mvs {
            let (i, j, pidx) = Self::tens_idx_from_mv(mv);
            let prob: f64 = self.0.double_value(&[i as i64, j as i64, pidx as i64]);
            probs.push(prob);
        }
        probs
    }

    pub fn to_prob(&self, mv: &Move) -> f64 {
        let (i, j, pidx) = Self::tens_idx_from_mv(mv);
        self.0.double_value(&[i as i64, j as i64, pidx as i64])
    }

    pub fn get_legal_inds(&self, legal_mvs: &Array<Move, 256>) -> Array<(i32, i32, i32), 256> {
        let mut inds: Array<(i32, i32, i32), 256> = Array::new();
        for mv in legal_mvs {
            let ind = Self::tens_idx_from_mv(mv);
            inds.push(ind);
        }
        inds
    }
}

impl Clone for PositionPrior {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PositionInferenceResult {
    priors: PositionPrior,
    value: f64,
    request_id: usize
}

impl PositionInferenceResult {
    pub async fn from_chessboard(chessboard: Chessboard) -> Self {
        todo!()
    }
    pub fn to_priors_and_value(self) -> (PositionPrior, f64) {
        (self.priors, self.value)
    }
    
    pub fn new(priors: PositionPrior, value: f64, request_id: usize) -> Self {
        Self { priors, value, request_id }
    }
    
    pub fn get_request_id(&self) -> usize {
        self.request_id
    }
}


pub struct PlaneMaskTensor(tch::Tensor);

impl PlaneMaskTensor {
    const EXPECTED_SHAPE: [i64; 2] = [8, 8];

    pub fn from_bitboard(board: u64) -> PlaneMaskTensor {
        let mut bits = [0i32; 64];
        for i in 0..64 {
            bits[i] = ((board >> i) as i32) & 1i32;
        }
        let mask: tch::Tensor = tch::Tensor::from_slice(&bits).view((8, 8)).flip(0).flip(1).to_kind(tch::Kind::Half);
        Self(mask)
    }

}

impl Default for PlaneMaskTensor {
    fn default() -> Self {
        Self(tch::Tensor::zeros(&Self::EXPECTED_SHAPE, (tch::Kind::Bool, tch::Device::Cpu)))
    }
}

pub struct PositionMetadataTensor(tch::Tensor);

impl PositionMetadataTensor {
    const EXPECTED_SHAPE: [i64; 3] = [8, 8, 7];

    pub fn new_zeros() -> Self {
        PositionMetadataTensor(tch::Tensor::zeros(&Self::EXPECTED_SHAPE, (tch::Kind::Half, tch::Device::Cpu)))
    }

    pub fn set_color(&mut self, color: Colors) {
        let _ = self.0.i((.., .., 0)).fill_(color as usize as f64);
    }

    pub fn set_total_move_count(&mut self, move_count: u32) {
        let _ = self.0.i((.., .., 1)).fill_(move_count as f64);
    }

    pub fn set_noprogress_count(&mut self, noprogress_count: i32) {
        let _ = self.0.i((.., .., 6)).fill_(noprogress_count as f64);
    }

    pub fn set_castling(&mut self, board: &Chessboard, p1_color: Colors) {
        let p1_can_castle_kingside = board.may_castle_kingside(p1_color) as u32 as f64;
        let p1_can_castle_queenside = board.may_castle_queenside(p1_color) as u32 as f64;
        let p2_can_castle_kingside = board.may_castle_kingside(Colors::opposite_color(p1_color)) as u32 as f64;
        let p2_can_castle_queenside = board.may_castle_queenside(Colors::opposite_color(p1_color)) as u32 as f64;
        let _ = self.0.i((.., .., 2)).fill_(p1_can_castle_kingside);
        let _ = self.0.i((.., .., 3)).fill_(p1_can_castle_queenside);
        let _ = self.0.i((.., .., 4)).fill_(p2_can_castle_kingside);
        let _ = self.0.i((.., .., 5)).fill_(p2_can_castle_queenside);
    }

}

pub struct PositionTensor(tch::Tensor);

impl PositionTensor {
    const EXPECTED_SHAPE: [i64; 3] = [8, 8, 14];

    pub fn set_plane_with_mask(&mut self, plane_index: usize, mask: &PlaneMaskTensor, value: f64) {
        let _ = self.0.i((.., .., plane_index as i64)).masked_fill_(&mask.0, value);
    }

    pub fn set_plane_with_bitboard(&mut self, plane_idx: usize, board: u64, value: f64) {
        let mask = PlaneMaskTensor::from_bitboard(board);
        self.set_plane_with_mask(plane_idx, &mask, value);
    }

    pub fn new_zeros() -> Self {
        PositionTensor(tch::Tensor::zeros(&Self::EXPECTED_SHAPE, (tch::Kind::Half, tch::Device::Cpu)))
    }
}

impl Default for PositionTensor {
    fn default() -> Self {
        Self(tch::Tensor::zeros(&Self::EXPECTED_SHAPE, (tch::Kind::Half, tch::Device::Cpu)))
    }
}

impl Borrow<tch::Tensor> for PositionTensor {
    fn borrow(&self) -> &tch::Tensor {
        &self.0
    }
}

pub struct PositionWithContextTensor(tch::Tensor);

impl PositionWithContextTensor {
    const EXPECTED_SHAPE: [i64; 3] = [8, 8, 119];

    pub fn new(mvs: Array<PositionTensor, 8>, meta: PositionMetadataTensor) -> Self {
        let all_mvs = tch::Tensor::cat(mvs.as_raw_ref(), 2);
        PositionWithContextTensor(tch::Tensor::cat(&[all_mvs, meta.0], 2))
    }

    pub fn into_tensor(self) -> tch::Tensor {
        self.0
    }
}

macro_rules! impl_serializable_tensor {
    (
        $name:ident,
        dtype = $kind:expr,
        rust_ty = $rust_ty:ty
    ) => {
        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let tensor = self.0.to_kind($kind).contiguous();

                let numel = tensor.numel();
                let mut data = vec![<$rust_ty as Default>::default(); numel as usize];

                tensor.copy_data(&mut data, numel);

                let bytes: &[u8] = bytemuck::cast_slice(&data);
                serializer.serialize_bytes(bytes)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct TensorVisitor;

                impl<'de> serde::de::Visitor<'de> for TensorVisitor {
                    type Value = $name;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("a byte buffer representing a tensor")
                    }

                    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        // SAFETY: bytes must be correctly aligned and sized
                        if v.len() % std::mem::size_of::<$rust_ty>() != 0 {
                            return Err(E::custom("invalid tensor byte length"));
                        }

                        let data: Vec<$rust_ty> =
                            bytemuck::cast_slice(v).to_vec();

                        let tensor = tch::Tensor::f_from_slice(&data)
                            .map_err(|e| E::custom(e.to_string()))?;

                        Ok($name(tensor))
                    }
                }

                deserializer.deserialize_bytes(TensorVisitor)
            }
        }
    };
}

impl_serializable_tensor!(PositionPrior, dtype = tch::Kind::Float, rust_ty = f32);
impl_serializable_tensor!(PositionWithContextTensor, dtype = tch::Kind::Float, rust_ty = f32);

#[derive(Serialize, Deserialize)]
pub struct PositionInferenceRequest {
    tensor: PositionWithContextTensor,
    move_id: usize,
    requester_id: uuid::Uuid,
}

impl PositionInferenceRequest {
    pub fn into_tuple(self) -> (PositionWithContextTensor, usize, uuid::Uuid) {
        (self.tensor, self.move_id, self.requester_id)
    }
}

