# ---------- build ----------
FROM rust:1.75-slim AS builder
WORKDIR /app


COPY . .
RUN cargo build --debug

# ---------- runtime ----------
FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/target/release/mcts_worker /usr/local/bin/mcts_worker

CMD ["mcts_worker"]
