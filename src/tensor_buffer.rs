// redis in background
// when queried for tensors, grabs a batch
// runs inference on a deployed model
// returns results to senders when inference completes
use std::sync::Arc;

use dashmap;
use flexbuffers;
use flexbuffers::FlexbufferSerializer;
use redis;
use redis::{AsyncCommands};
use serde::{Deserialize, Serialize};
use tokio;
use uuid;

use crate::consts::*;
use crate::inference_primitives::*;

fn serialize<T>(value: T) -> FlexbufferSerializer where T: Serialize {
    let mut s = flexbuffers::FlexbufferSerializer::new();
    value.serialize(&mut s).unwrap();
    s
}

pub struct InferenceClient {
    connection: redis::aio::MultiplexedConnection,
    pending: Arc<dashmap::DashMap<usize, tokio::sync::oneshot::Sender<PositionInferenceResult>>>,
    counter: std::sync::atomic::AtomicUsize,
    client_id: uuid::Uuid,
}

impl InferenceClient {
    pub fn new(uri: &str, rt: &tokio::runtime::Runtime) -> Self {
        let connection = rt.block_on(async {
            let client = redis::Client::open(uri)?;
            client.get_multiplexed_async_connection().await
        }).unwrap();
        let pending = Arc::new(dashmap::DashMap::new());
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let client_id = uuid::Uuid::new_v4();
        // TODO spawn listener immediately
        Self { connection, pending, counter, client_id  }
    }

    async fn listen_for_inference_results(&self) {
        let mut connection = self.connection.clone();
        loop {
            let (_key, payload): (String, Vec<u8>) = connection.brpop(self.client_id.to_string(), 0.0).await.unwrap();

            let r = flexbuffers::Reader::get_root(payload.as_slice()).unwrap();
            let result = PositionInferenceResult::deserialize(r).unwrap();

            let rx = self.pending.remove(&result.get_request_id()).unwrap().1;
            rx.send(result).unwrap();
        }
    }

    pub async fn send_tensor(&self, tens: PositionWithContextTensor) -> (usize, tokio::sync::oneshot::Receiver<PositionInferenceResult>) {
        let mut connection = self.connection.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let id = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let buf = serialize(tens);

        let _: () = connection.rpush(INFERENCE_BUFFER, buf.view()).await.unwrap();
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
    // TODO handle disconnected clients; clean up their lists
    pub fn new(uri: &str, minimum_batch_size: usize, max_poll_misses: usize, short_poll_duration: f64, rt: &'a tokio::runtime::Runtime) -> Self {
        let connection = rt.block_on(async {
            let client = redis::Client::open(uri)?;
            client.get_multiplexed_async_connection().await
        }).unwrap();
        Self { connection, minimum_batch_size, max_poll_misses, short_poll_duration, rt }
    }

    async fn get_batch(&self) -> (tch::Tensor, Vec<usize>, Vec<uuid::Uuid>) {
        let mut connection = self.connection.clone();
        let mut tens_buf: Vec<tch::Tensor> = Vec::new();
        let mut move_id_buf: Vec<usize> = Vec::new();
        let mut request_id_buf: Vec<uuid::Uuid> = Vec::new();
        // TODO tokio run with timeout that returns whatever we got
        let mut count_misses = 0;
        while (tens_buf.len() < self.minimum_batch_size && count_misses < self.max_poll_misses) || (tens_buf.len() == 0) {
            let rr: redis::RedisResult<(String, Vec<u8>)> = connection.blpop(INFERENCE_BUFFER, self.short_poll_duration).await;
            match rr {
                Ok((_key, payload)) => {
                    let r = flexbuffers::Reader::get_root(payload.as_slice()).unwrap();
                    let (result, move_id, requester_id) = PositionInferenceRequest::deserialize(r).unwrap().into_tuple();
                    tens_buf.push(result.into_tensor());
                    move_id_buf.push(move_id);
                    request_id_buf.push(requester_id);
                }
                Err(_) => {
                    if tens_buf.len() > 0 {
                        count_misses += 1;
                    }
                }
            }
        }
        (tch::Tensor::stack(&tens_buf, 0), move_id_buf, request_id_buf)
    }

    pub fn python_get_batch(&self) -> (tch::Tensor, Vec<usize>, Vec<uuid::Uuid>) {
        self.rt.block_on(self.get_batch())
    }

    pub async fn publish_inference_results(&self, stacked_priors: tch::Tensor, value: Vec<f64>, request_ids: Vec<usize>, client_ids: Vec<uuid::Uuid>) {
        let mut connection = self.connection.clone();

        debug_assert_eq!(stacked_priors.size()[0] as usize, value.len());
        debug_assert_eq!(value.len(), request_ids.len());
        debug_assert_eq!(request_ids.len(), client_ids.len());
        let n = value.len();
        for i in 0..n {
            let result = PositionInferenceResult::new(PositionPrior::from_tensor(stacked_priors.get(i as i64)), value[i], request_ids[i]);
            let buf = serialize(result);
            let _: () = connection.rpush(client_ids[i].to_string(), buf.view()).await.unwrap(); // TODO could be bad
        }
    }
}
