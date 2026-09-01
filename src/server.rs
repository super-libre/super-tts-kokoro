// SPDX-License-Identifier: GPL-3.0-only
//! The `/v1` contract, served over the Unix socket the daemon hands us.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::frames;
use crate::model::{Kokoro, SAMPLE_RATE, split_for_streaming};

/// Where the model is in its lifecycle, as `GET /v1/status` reports it.
#[derive(Debug, Clone)]
enum LoadState {
    /// Serving, but no model has been asked for yet.
    Starting,
    /// A `POST /v1/load` is in flight.
    Loading,
    /// Ready to synthesize.
    Ready { model: String, device: String },
    /// The load failed; the reason reaches the user through the daemon.
    Failed { reason: String },
}

/// Everything the handlers share.
pub struct AppState {
    backend_dir: PathBuf,
    /// The loaded model. A `std::sync::Mutex` rather than a `tokio` one
    /// because every use is inside `spawn_blocking`: ONNX Runtime needs `&mut`
    /// and blocks, so it must never be touched from the async runtime.
    model: Mutex<Option<Kokoro>>,
    state: RwLock<LoadState>,
    /// Set by `POST /v1/cancel`, cleared at the start of each synthesis.
    cancelled: AtomicBool,
}

impl AppState {
    /// State for a backend rooted at `backend_dir`.
    #[must_use]
    pub fn new(backend_dir: PathBuf) -> Self {
        Self {
            backend_dir,
            model: Mutex::new(None),
            state: RwLock::new(LoadState::Starting),
            cancelled: AtomicBool::new(false),
        }
    }

    fn set_state(&self, next: LoadState) {
        // A poisoned lock means a handler panicked mid-update. The state is a
        // plain enum with no invariant to repair, so recovering is correct and
        // keeps one panic from taking the backend down with it.
        let mut guard = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = next;
    }

    fn state(&self) -> LoadState {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// Build the router.
pub fn router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(status))
        .route("/v1/load", post(load))
        .route("/v1/synthesize", post(synthesize))
        .route("/v1/cancel", post(cancel))
        .layer(axum::extract::DefaultBodyLimit::max(1 << 20))
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

async fn status(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(match s.state() {
        LoadState::Starting => json!({ "status": "success", "state": "starting" }),
        LoadState::Loading => json!({ "status": "success", "state": "loading" }),
        LoadState::Ready { model, device } => json!({
            "status": "success",
            "state": "ready",
            "device": device,
            "model": { "name": model },
        }),
        LoadState::Failed { reason } => json!({
            "status": "error",
            "state": "error",
            "reason": reason,
        }),
    })
}

/// `POST /v1/load` body. `provider` is accepted and ignored — it is a
/// compatibility echo from an older identity scheme, and the contract says a
/// new backend should not validate it.
#[derive(Debug, Deserialize)]
struct LoadRequest {
    name: String,
    #[serde(default)]
    device: Option<String>,
}

/// `POST /v1/load` — answer `202` immediately and load in the background.
///
/// The daemon polls `GET /v1/status` for readiness, so returning before the
/// model is resident is the contract rather than a shortcut: a session build
/// over 325 MB of weights takes long enough to time out an HTTP request.
async fn load(State(s): State<Arc<AppState>>, body: Option<Json<LoadRequest>>) -> Response {
    let Some(Json(req)) = body else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "malformed load body" })),
        )
            .into_response();
    };

    s.set_state(LoadState::Loading);
    let device = req.device.unwrap_or_else(|| "cpu".to_string());
    let state = Arc::clone(&s);
    let name = req.name.clone();

    tokio::task::spawn_blocking(move || {
        match Kokoro::load(&state.backend_dir, &name) {
            Ok(model) => {
                *state
                    .model
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(model);
                // Kokoro runs on the CPU here: the session is built with the
                // default execution provider, so reporting the requested
                // device would tell the user something untrue.
                if device != "cpu" {
                    log::info!("device {device:?} requested; this build runs on the CPU");
                }
                state.set_state(LoadState::Ready {
                    model: name,
                    device: "cpu".to_string(),
                });
            }
            Err(e) => {
                log::error!("load failed: {e:#}");
                state.set_state(LoadState::Failed {
                    reason: format!("{e:#}"),
                });
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
        .into_response()
}

async fn cancel(State(s): State<Arc<AppState>>) -> Json<Value> {
    s.cancelled.store(true, Ordering::SeqCst);
    Json(json!({ "status": "success", "message": "Cancelled" }))
}

/// `POST /v1/synthesize` body.
#[derive(Debug, Deserialize)]
struct SynthesizeRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

/// `POST /v1/synthesize` — stream framed audio as each segment finishes.
async fn synthesize(
    State(s): State<Arc<AppState>>,
    body: Option<Json<SynthesizeRequest>>,
) -> Response {
    let Some(Json(req)) = body else {
        return json_error(StatusCode::BAD_REQUEST, "malformed synthesize body");
    };
    if req.text.trim().is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "text is empty");
    }
    if !matches!(s.state(), LoadState::Ready { .. }) {
        return json_error(StatusCode::CONFLICT, "not_ready");
    }

    // Resolve the voice before streaming starts. Once the response is a 200 the
    // only way to report a bad voice is an error frame, which the daemon
    // surfaces as a synthesis failure rather than a bad request.
    let voice = {
        let guard = s
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(model) = guard.as_ref() else {
            return json_error(StatusCode::CONFLICT, "not_ready");
        };
        match req.voice.as_deref() {
            None => model.default_voice().to_string(),
            Some(v) if model.has_voice(v) => v.to_string(),
            Some(v) => return json_error(StatusCode::BAD_REQUEST, &format!("unknown voice {v}")),
        }
    };

    s.cancelled.store(false, Ordering::SeqCst);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let state = Arc::clone(&s);
    let speed = req.speed.unwrap_or(1.0).clamp(0.5, 2.0);
    let language = req.language.clone();
    let text = req.text.clone();

    tokio::task::spawn_blocking(move || {
        stream_synthesis(&state, &text, &voice, language.as_deref(), speed, &tx);
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.super-tts.frames")
        .header(HeaderName::from_static("x-tts-sample-rate"), SAMPLE_RATE)
        .header(HeaderName::from_static("x-tts-channels"), 1)
        .header(HeaderName::from_static("x-tts-format"), "s16le")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .unwrap_or_else(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))
}

/// Synthesize segment by segment, sending frames as each completes.
///
/// Runs on a blocking thread. Every exit path sends a terminal frame: the
/// daemon treats a stream that stops without one as a failure, which is exactly
/// right for a crash but would be wrong for the ordinary end of an utterance.
fn stream_synthesis(
    state: &Arc<AppState>,
    text: &str,
    voice: &str,
    language: Option<&str>,
    speed: f32,
    tx: &tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
) {
    let send = |bytes: Vec<u8>| tx.blocking_send(Ok(bytes::Bytes::from(bytes))).is_ok();

    let mut guard = state
        .model
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(model) = guard.as_mut() else {
        let _ = send(frames::error_frame("model not loaded"));
        return;
    };

    let mut elapsed_ms = 0_u64;
    let mut char_offset = 0_u32;
    let mut spoke_anything = false;
    let mut cancelled = false;

    for segment in split_for_streaming(text) {
        if state.cancelled.load(Ordering::SeqCst) {
            log::info!("synthesis cancelled");
            cancelled = true;
            break;
        }
        let chars = u32::try_from(segment.chars().count()).unwrap_or(u32::MAX);
        let samples = match model.synthesize_segment(&segment, voice, language, speed) {
            Ok(s) => s,
            Err(e) => {
                log::error!("synthesis failed: {e:#}");
                let _ = send(frames::error_frame(&format!("{e:#}")));
                return;
            }
        };
        // A segment of pure punctuation synthesizes nothing. Skip it and keep
        // the character offset moving, so later marks still point at the right
        // span of the request text.
        if samples.is_empty() {
            char_offset = char_offset.saturating_add(chars);
            continue;
        }
        spoke_anything = true;

        let duration_ms = (samples.len() as u64 * 1000) / u64::from(SAMPLE_RATE);
        for frame in frames::audio_frames(&samples) {
            if !send(frame) {
                // The daemon hung up — a cancelled utterance, usually.
                log::debug!("receiver dropped; abandoning synthesis");
                return;
            }
        }
        if !send(frames::mark_frame(
            elapsed_ms,
            elapsed_ms + duration_ms,
            char_offset,
            char_offset.saturating_add(chars),
        )) {
            return;
        }
        elapsed_ms += duration_ms;
        char_offset = char_offset.saturating_add(chars);
    }

    // Nothing pronounceable in the whole request. A `done` here would report a
    // successful utterance the user never heard, so it is a failure — but only
    // once every segment has had its turn, never because one of them was a
    // comma. A cancelled utterance is not this: it stopped on request.
    if !spoke_anything && !cancelled {
        let _ = send(frames::error_frame("text contained nothing to speak"));
        return;
    }

    let _ = send(frames::done_frame());
}

fn json_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({ "status": "error", "message": message })),
    )
        .into_response()
}
