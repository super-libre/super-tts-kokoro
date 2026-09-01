// SPDX-License-Identifier: GPL-3.0-only
//! Kokoro-82M text-to-speech backend for Super TTS.
//!
//! A subprocess backend: the daemon spawns this binary inside a hardened
//! `systemd-run --user` transient unit and drives the `/v1` contract over a
//! Unix socket. Two environment variables are the whole interface —
//! `SUPER_TTS_BACKEND_SOCKET` names the socket to bind, `SUPER_TTS_BACKEND_DIR`
//! the directory holding `backend.toml` and the downloaded weights.
//!
//! The sandbox shapes the design more than anything else: there is no network
//! (`PrivateNetwork=yes`) and the backend directory is mounted read-only, so
//! this process never downloads or writes a model file. The daemon fetches
//! everything named in `[[models.files]]` before the first `POST /v1/load`.

mod en;
mod en_numbers;
mod en_tagger;
mod en_tokenizer;
mod espeak;
mod frames;
mod ja;
mod ja_numbers;
mod model;
mod phonemes;
mod server;
mod voices;
mod zh;
mod zh_numbers;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::net::UnixListener;

/// Names the Unix socket to bind.
const ENV_SOCKET: &str = "SUPER_TTS_BACKEND_SOCKET";
/// Names the backend's own directory.
const ENV_DIR: &str = "SUPER_TTS_BACKEND_DIR";

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let socket_path = PathBuf::from(
        std::env::var(ENV_SOCKET).with_context(|| format!("{ENV_SOCKET} is not set"))?,
    );
    let backend_dir =
        PathBuf::from(std::env::var(ENV_DIR).with_context(|| format!("{ENV_DIR} is not set"))?);

    // The socket directory is the one path the sandbox leaves writable, and a
    // stale socket from a killed unit would make `bind` fail with EADDRINUSE.
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("binding {}", socket_path.display()))?;
    log::info!(
        "kokoro backend listening on {} (dir {})",
        socket_path.display(),
        backend_dir.display()
    );

    let state = Arc::new(server::AppState::new(backend_dir));
    let app = server::router(state);

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                log::error!("accept failed: {e}");
                continue;
            }
        };
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}
