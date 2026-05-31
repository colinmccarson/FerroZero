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

RUN cargo build --bin worker

# Copy downloaded libtorch to a stable location
RUN mkdir -p /opt/libtorch && \
    cp -r \
    $(find /app/target/debug/build -path "*/out/libtorch/libtorch" -type d | head -1)/* \
    /opt/libtorch

# ---------- runtime ----------
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /app/target/debug/worker /usr/local/bin/worker

ENV LD_LIBRARY_PATH=/opt/libtorch/lib

CMD ["worker"]