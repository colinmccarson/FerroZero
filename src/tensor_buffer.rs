// redis in background
// when queried for tensors, grabs a batch
// runs inference on a deployed model
// returns results to senders when inference completes
use std::sync::Arc;

use dashmap;
use flexbuffers;
use redis;
use redis::{AsyncCommands};
use serde::{Deserialize, Serialize};
use tokio;

use crate::consts::*;
use crate::inference_primitives::*;

pub struct InferenceClient {
    connection: redis::aio::MultiplexedConnection,
    pending: Arc<dashmap::DashMap<usize, tokio::sync::oneshot::Sender<PositionInferenceResult>>>,
    counter: std::sync::atomic::AtomicUsize,
}

impl InferenceClient {
    pub fn new(uri: &str, rt: &tokio::runtime::Runtime) -> Self {
        let connection = rt.block_on(async {
            let client = redis::Client::open(uri)?;
            client.get_multiplexed_async_connection().await
        }).unwrap();
        let pending = Arc::new(dashmap::DashMap::new());
        let counter = std::sync::atomic::AtomicUsize::new(0);
        // TODO spawn listener immediately
        Self { connection, pending, counter }
    }

    async fn listen_for_inference_results(&self) {
        let mut connection = self.connection.clone();
        loop {
            let (_key, payload): (String, Vec<u8>) = connection.brpop(INFERENCE_RESULT_BUFFER, 0.0).await.unwrap();

            let r = flexbuffers::Reader::get_root(payload.as_slice()).unwrap();
            let result = SerializableInferenceResult::deserialize(r).unwrap();

            let (position_inference_result, id) = result.into_inference_result_and_id();

            let mut rx = self.pending.get(&id).unwrap();
            rx.send(position_inference_result).unwrap();
            self.pending.remove(&id);
        }
    }

    pub async fn send_tensor(&self, tens: PositionWithContextTensor) -> (usize, tokio::sync::oneshot::Receiver<PositionInferenceResult>) {
        let mut connection = self.connection.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let id = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut s = flexbuffers::FlexbufferSerializer::new();
        let serializable = tens.to_serializable(id);
        serializable.serialize(&mut s).unwrap();

        let _: () = connection.lpush(INFERENCE_BUFFER, s.view()).await.unwrap();
        self.pending.insert(id, tx); // dashmap magic
        (id, rx)
    }

}


pub struct InferenceBatchManager<'a> {
    connection: redis::aio::MultiplexedConnection,
    minimum_batch_size: usize,
    max_poll_misses: usize,
    short_poll_duration: f64,
    rt: &'a tokio::runtime::Runtime,
}

impl<'a> InferenceBatchManager<'a> {
    pub fn new(uri: &str, minimum_batch_size: usize, max_poll_misses: usize, short_poll_duration: f64, rt: &'a tokio::runtime::Runtime) -> Self {
        let connection = rt.block_on(async {
            let client = redis::Client::open(uri)?;
            client.get_multiplexed_async_connection().await
        }).unwrap();
        Self { connection, minimum_batch_size, max_poll_misses, short_poll_duration, rt }
    }

    async fn get_batch(&self) -> (tch::Tensor, Vec<usize>) { // TODO get rid of async
        let mut connection = self.connection.clone();
        let mut tens_buf: Vec<tch::Tensor> = Vec::new();
        let mut id_buf: Vec<usize> = Vec::new();
        // TODO tokio run with timeout that returns whatever we got
        let mut count_misses = 0;
        while (tens_buf.len() < self.minimum_batch_size && count_misses < self.max_poll_misses) || (tens_buf.len() == 0) {
            let rr: redis::RedisResult<(String, Vec<u8>)> = connection.blpop(INFERENCE_BUFFER, self.short_poll_duration).await;
            match rr {
                Ok((_key, payload)) => {
                    let r = flexbuffers::Reader::get_root(payload.as_slice()).unwrap();
                    let (result, id) = SerializedPositionWithContextAndId::deserialize(r).unwrap().into_tuple();
                    tens_buf.push(result.into_tensor());
                    id_buf.push(id);
                }
                Err(_) => {
                    if tens_buf.len() > 0 {
                        count_misses += 1;
                    }
                }
            }
        }
        (tch::Tensor::stack(&tens_buf, 0), id_buf)
    }

    pub fn python_get_batch(&self) -> (tch::Tensor, Vec<usize>) {
        self.rt.block_on(self.get_batch())
    }
}
