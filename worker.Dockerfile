# ---------- build ----------
FROM rust:1.89-slim AS builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        pkg-config \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build

# ---------- runtime ----------
FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/target/debug/worker /usr/local/bin/worker

CMD ["worker"]
