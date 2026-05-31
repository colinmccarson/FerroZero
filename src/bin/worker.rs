use std::sync::Arc;

use tch;
use tokio;

use alphazero::consts;
use alphazero::inference_primitives::*;
use alphazero::tensor_buffer::*;
use alphazero::tree_search::*;


async fn mock_inference(mut batch_manager: InferenceBatchManager, mut shutdown: tokio::sync::watch::Receiver<()>) {
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            (_, move_ids, requester_ids) = batch_manager.get_batch() => {
                let mut priors: Vec<tch::Tensor> = Vec::new();
                let volume = PositionPrior::EXPECTED_SHAPE[0] * PositionPrior::EXPECTED_SHAPE[1] * PositionPrior::EXPECTED_SHAPE[2];
                let unif_prob: f64 = 1.0 / (volume as f64);
                let _ = move_ids.iter().map(|_| {
                    let tens = tch::Tensor::ones(&PositionPrior::EXPECTED_SHAPE, (tch::Kind::Float, tch::Device::Cpu));
                    priors.push(tens * unif_prob);
                });

                let result = tch::Tensor::stack(&priors, 0);
                let values: Vec<f64> = move_ids.iter().map(|_| { 0.0 }).collect();

                batch_manager.publish_inference_results(result, values, move_ids, requester_ids).await;
            }
        }
    }
}


fn main() {
    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());

    let batch_manager = InferenceBatchManager::new(
        consts::INFERENCE_BUFFER, 1, 1, 1.0, rt.handle().clone(),
    );
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());
    rt.spawn(async move {
        mock_inference(batch_manager, shutdown_rx).await;
    });

    let mut chessgame = ChessTree::new_with_inference(0.5, rt.handle().clone());
    chessgame.simulate(100);

    shutdown_tx.send(()).unwrap();
}
