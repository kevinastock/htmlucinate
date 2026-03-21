default: build

build:
    cargo build

run *ARGS:
    cargo run -- {{ARGS}}

test:
    cargo test

clippy:
    cargo clippy -- -D warnings

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

check: fmt-check clippy test
