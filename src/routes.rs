use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use rig::completion::message::Message;
use serde::Deserialize;
use tokio_stream::StreamExt;
use tracing::info;
use uuid::Uuid;

use crate::AppState;
use crate::agent;
use crate::session::Session;

// ---------------------------------------------------------------------------
// Default new-tab image (generated at startup)
// ---------------------------------------------------------------------------

fn generate_default_image() -> Vec<u8> {
    let width = 1536u32;
    let height = 1024u32;
    let mut img = image::RgbaImage::new(width, height);

    // Light neutral background with a subtle warm gradient
    for y in 0..height {
        for x in 0..width {
            let r = (245.0 - 10.0 * (y as f64 / height as f64)) as u8;
            let g = (245.0 - 8.0 * (y as f64 / height as f64)) as u8;
            let b = (248.0 - 5.0 * (y as f64 / height as f64)) as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }

    // Draw a centered card
    let card_x = width / 4;
    let card_y = height / 4;
    let card_w = width / 2;
    let card_h = height / 2;
    for y in card_y..(card_y + card_h) {
        for x in card_x..(card_x + card_w) {
            img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
        }
    }

    // Card border — light gray
    for x in card_x..(card_x + card_w) {
        img.put_pixel(x, card_y, image::Rgba([200, 200, 200, 255]));
        img.put_pixel(x, card_y + card_h - 1, image::Rgba([200, 200, 200, 255]));
    }
    for y in card_y..(card_y + card_h) {
        img.put_pixel(card_x, y, image::Rgba([200, 200, 200, 255]));
        img.put_pixel(card_x + card_w - 1, y, image::Rgba([200, 200, 200, 255]));
    }

    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("Failed to encode default image");
    buf.into_inner()
}

static DEFAULT_IMAGE: std::sync::LazyLock<Arc<Vec<u8>>> =
    std::sync::LazyLock::new(|| Arc::new(generate_default_image()));

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/api/session", post(create_session))
        .route("/api/session/{id}/query", post(handle_query))
        .route("/api/session/{id}/click", post(handle_click))
        .route("/api/session/{id}/pagedown", post(handle_pagedown))
        .route("/api/session/{id}/back", post(handle_back))
        .route("/api/session/{id}/image", get(get_image))
        .route("/api/session/{id}/log", get(stream_log))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// HTML page
// ---------------------------------------------------------------------------

async fn index_page() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct QueryPayload {
    query: String,
    viewport_width: Option<u32>,
    viewport_height: Option<u32>,
}

#[derive(Deserialize)]
struct ClickPayload {
    x: f64, // 0.0 - 1.0 relative position
    y: f64,
    viewport_width: Option<u32>,
    viewport_height: Option<u32>,
}

#[derive(Deserialize)]
struct PagedownPayload {
    viewport_width: Option<u32>,
    viewport_height: Option<u32>,
}

#[derive(serde::Serialize)]
struct ActionResponse {
    url: String,
    /// Base64-encoded PNG image
    image: String,
}

#[derive(serde::Serialize)]
struct SessionResponse {
    session_id: Uuid,
    url: String,
    image: String,
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn create_session(State(state): State<Arc<AppState>>) -> Json<SessionResponse> {
    let id = Uuid::new_v4();
    let session = Session::new(DEFAULT_IMAGE.clone());
    let image_b64 = BASE64.encode(session.current_image.as_ref());
    state.sessions.insert(id, session);
    info!("Created session {id}");
    Json(SessionResponse {
        session_id: id,
        url: String::new(),
        image: image_b64,
    })
}

async fn handle_query(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<QueryPayload>,
) -> Result<Json<ActionResponse>, AppError> {
    let (history, log_tx, viewport) = {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        if let (Some(w), Some(h)) = (payload.viewport_width, payload.viewport_height) {
            session.viewport = (w, h);
        }
        session.push_snapshot();
        (
            rig_history_from_session(&session),
            session.log_tx.clone(),
            session.viewport,
        )
    };

    let user_msg = Message::user(payload.query.clone());

    let (result, new_history) = agent::run_agent(
        &state.openai_client,
        &state.args.agent_model,
        &state.args.image_model,
        viewport,
        log_tx,
        history,
        user_msg,
    )
    .await?;

    // Update session
    let image_b64 = BASE64.encode(&result.image_png);
    {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        session.current_image = Arc::new(result.image_png);
        session.omnibar_url = result.url.clone();
        save_history_to_session(&mut session, &new_history);
    }

    Ok(Json(ActionResponse {
        url: result.url,
        image: image_b64,
    }))
}

async fn handle_click(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<ClickPayload>,
) -> Result<Json<ActionResponse>, AppError> {
    let (history, log_tx, viewport, click_image) = {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        if let (Some(w), Some(h)) = (payload.viewport_width, payload.viewport_height) {
            session.viewport = (w, h);
        }
        session.push_snapshot();

        // Draw click circle on current image
        let click_img = agent::draw_click_circle(&session.current_image, payload.x, payload.y)
            .map_err(|e| AppError::Internal(format!("Failed to draw click circle: {e}")))?;

        // Save click-annotated image to debug/ for inspection
        let debug_dir = std::path::Path::new("debug");
        if std::fs::create_dir_all(debug_dir).is_ok() {
            let filename = format!(
                "click_{id}_{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
            if let Err(e) = std::fs::write(debug_dir.join(&filename), &click_img) {
                tracing::warn!("Failed to save debug click image: {e}");
            } else {
                tracing::info!("Saved click image to debug/{filename}");
            }
        }

        (
            rig_history_from_session(&session),
            session.log_tx.clone(),
            session.viewport,
            click_img,
        )
    };

    // Build message with image + text
    let user_msg = agent::user_message_with_image(
        &format!(
            "The user clicked at position ({:.1}%, {:.1}%) on the current page. \
             The red circle on the attached screenshot shows where they clicked. \
             Interpret what they clicked on and produce the resulting page.",
            payload.x * 100.0,
            payload.y * 100.0
        ),
        &click_image,
    );

    let (result, new_history) = agent::run_agent(
        &state.openai_client,
        &state.args.agent_model,
        &state.args.image_model,
        viewport,
        log_tx,
        history,
        user_msg,
    )
    .await?;

    let image_b64 = BASE64.encode(&result.image_png);
    {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        session.current_image = Arc::new(result.image_png);
        session.omnibar_url = result.url.clone();
        save_history_to_session(&mut session, &new_history);
    }

    Ok(Json(ActionResponse {
        url: result.url,
        image: image_b64,
    }))
}

async fn handle_pagedown(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<PagedownPayload>,
) -> Result<Json<ActionResponse>, AppError> {
    let (history, log_tx, viewport) = {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        if let (Some(w), Some(h)) = (payload.viewport_width, payload.viewport_height) {
            session.viewport = (w, h);
        }
        session.push_snapshot();
        (
            rig_history_from_session(&session),
            session.log_tx.clone(),
            session.viewport,
        )
    };

    let user_msg = Message::user(
        "The user pressed Page Down. Scroll the current page down by one viewport height \
         and show the next screenful of content via update_page.",
    );

    let (result, new_history) = agent::run_agent(
        &state.openai_client,
        &state.args.agent_model,
        &state.args.image_model,
        viewport,
        log_tx,
        history,
        user_msg,
    )
    .await?;

    let image_b64 = BASE64.encode(&result.image_png);
    {
        let mut session = state
            .sessions
            .get_mut(&id)
            .ok_or(AppError::SessionNotFound)?;
        session.current_image = Arc::new(result.image_png);
        session.omnibar_url = result.url.clone();
        save_history_to_session(&mut session, &new_history);
    }

    Ok(Json(ActionResponse {
        url: result.url,
        image: image_b64,
    }))
}

async fn handle_back(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ActionResponse>, AppError> {
    let mut session = state
        .sessions
        .get_mut(&id)
        .ok_or(AppError::SessionNotFound)?;
    if !session.go_back() {
        return Err(AppError::Internal("No history to go back to".into()));
    }
    Ok(Json(ActionResponse {
        url: session.omnibar_url.clone(),
        image: BASE64.encode(session.current_image.as_ref()),
    }))
}

async fn get_image(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Response, AppError> {
    let session = state.sessions.get(&id).ok_or(AppError::SessionNotFound)?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "image/png")],
        session.current_image.as_ref().clone(),
    )
        .into_response())
}

async fn stream_log(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, AppError>
{
    let rx = {
        let session = state.sessions.get(&id).ok_or(AppError::SessionNotFound)?;
        session.log_tx.subscribe()
    };

    let stream = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|item| match item {
        Ok(entry) => {
            let data = serde_json::to_string(&entry).unwrap_or_default();
            Some(Ok(Event::default().data(data)))
        }
        Err(_) => None,
    });

    Ok(Sse::new(stream))
}

// ---------------------------------------------------------------------------
// History serialization
// ---------------------------------------------------------------------------

/// Convert session's stored JSON messages back to rig Messages.
fn rig_history_from_session(
    session: &dashmap::mapref::one::RefMut<'_, Uuid, Session>,
) -> Vec<Message> {
    session
        .messages
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect()
}

fn save_history_to_session(
    session: &mut dashmap::mapref::one::RefMut<'_, Uuid, Session>,
    history: &[Message],
) {
    session.messages = history
        .iter()
        .filter_map(|m| serde_json::to_value(m).ok())
        .collect();
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

enum AppError {
    SessionNotFound,
    Internal(String),
    Agent(agent::AgentRunError),
}

impl From<agent::AgentRunError> for AppError {
    fn from(e: agent::AgentRunError) -> Self {
        AppError::Agent(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::SessionNotFound => (StatusCode::NOT_FOUND, "Session not found".to_string()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::Agent(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
        (status, Json(ErrorResponse { error: msg })).into_response()
    }
}
