use std::sync::Arc;

use image::ImageReader;
use rig::OneOrMany;
use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::client::CompletionClient;
use rig::completion::message::{
    AssistantContent, DocumentSourceKind, Image, ImageMediaType, Message, Text, UserContent,
};
use rig::completion::{CompletionModel, CompletionResponse, Prompt, ToolDefinition};
use rig::image_generation::ImageGenerationModel;
use rig::prelude::ImageGenerationClient;
use rig::providers::openai;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::session::LogEntry;

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are the rendering engine of a web browser called "HTMLucinate". Your job is to produce accurate visual renderings of web pages exactly as a real browser would display them.

When the user types a URL or search query into the omnibar:
1. Figure out what web page they are trying to visit. If you need to search, use
   fetch_url with the DuckDuckGo API: https://api.duckduckgo.com/?q=YOUR+QUERY&format=json&no_html=1
2. Fetch the real page with fetch_url to get its actual content, layout, and structure.
3. Call update_page with a detailed visual description of how the page should be rendered.

When the user clicks somewhere on the current page, you'll receive the screenshot with
a red circle indicating where they clicked. Interpret the click (link, button, etc.)
and produce the resulting page via update_page.

When the user asks to scroll down ("page down"), produce the next screenful of content
via update_page.

Important guidelines:
- Always call update_page exactly once per user action.
- Always fetch the real page with fetch_url before rendering. Base your rendering on the actual page content — do not guess or invent content.
- The page_description should be a faithful, detailed visual description of ONLY the web page content area as it would appear in a browser viewport. Do NOT describe browser chrome (address bar, tabs, back/forward buttons, window frame) — that is handled externally.
- Render the page accurately: use the real text, real layout, real colors, and real structure from the fetched content. Your goal is to match what a real browser would show as closely as possible.
- Do NOT add, remove, or modify page content. Do not inject humor, easter eggs, or fictional elements. Render what is actually there.
- Include all visible page elements: navigation bars, sidebars, footers, ads, banners, images, etc. — but only those that are actually present in the page content.
"#;

// ---------------------------------------------------------------------------
// Logging hook
// ---------------------------------------------------------------------------

/// Sends concise "status" messages (shown in the loading overlay) and detailed
/// "tool_call" / "tool_result" / "assistant" messages (shown in the log modal).
#[derive(Clone)]
pub struct LogHook {
    pub tx: broadcast::Sender<LogEntry>,
}

impl LogHook {
    /// Send a short status update shown to the user in the loading overlay.
    fn status(&self, msg: impl Into<String>) {
        let _ = self.tx.send(LogEntry {
            kind: "status".into(),
            content: msg.into(),
        });
    }
    /// Send a detailed log entry shown in the log modal.
    fn log(&self, kind: &str, content: impl Into<String>) {
        let _ = self.tx.send(LogEntry {
            kind: kind.into(),
            content: content.into(),
        });
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max])
    } else {
        s.to_string()
    }
}

impl<M: CompletionModel> PromptHook<M> for LogHook {
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        info!(tool = tool_name, "Tool call");
        tracing::debug!(tool = tool_name, args, "Tool call args");

        // Concise status for loading overlay
        let status = match tool_name {
            "fetch_url" => {
                let u = serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("url").and_then(|u| u.as_str()).map(String::from))
                    .unwrap_or_default();
                format!("Fetching: {}", truncate(&u, 60))
            }
            "update_page" => "Generating page image…".into(),
            _ => format!("Calling {tool_name}…"),
        };
        self.status(&status);
        self.log("tool", format!("{tool_name}({}", truncate(args, 200)));
        ToolCallHookAction::Continue
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        info!(tool = tool_name, result_len = result.len(), "Tool result");
        tracing::debug!(
            tool = tool_name,
            result = truncate(result, 500).as_str(),
            "Tool result content"
        );

        match tool_name {
            "update_page" => self.status("Image generated"),
            "fetch_url" => self.status(format!("Fetched {} bytes", result.len())),
            _ => self.status(format!("{tool_name} done")),
        }
        self.log("result", format!("{tool_name} → {}", truncate(result, 300)));
        HookAction::Continue
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        response: &CompletionResponse<M::Response>,
    ) -> HookAction {
        let mut texts = Vec::new();
        let mut tool_calls = Vec::new();
        for item in response.choice.iter() {
            match item {
                AssistantContent::Text(t) => texts.push(t.text.clone()),
                AssistantContent::ToolCall(tc) => {
                    tool_calls.push(tc.function.name.clone());
                }
                _ => {}
            }
        }
        if !texts.is_empty() {
            let text = texts.join(" ");
            let text_len = text.len();
            info!(text_len, "Assistant text response");
            tracing::debug!(
                text = truncate(&text, 500).as_str(),
                "Assistant text content"
            );
            self.log("agent", truncate(&text, 300));
        }
        if !tool_calls.is_empty() {
            let calls = tool_calls.join(", ");
            info!(calls = calls.as_str(), "Assistant tool calls");
            self.status(format!("Agent using: {calls}"));
        }
        if texts.is_empty() && tool_calls.is_empty() {
            info!("Assistant responded with no text or tool calls");
        }
        HookAction::Continue
    }

    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        let prompt_summary = match prompt {
            Message::User { content } => {
                let parts: Vec<String> = content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text(t) => truncate(&t.text, 100),
                        UserContent::Image(_) => "[image]".to_string(),
                        _ => "[other]".to_string(),
                    })
                    .collect();
                parts.join(" ")
            }
            _ => "system/assistant message".to_string(),
        };
        info!(
            history_len = history.len(),
            prompt = prompt_summary.as_str(),
            "Sending completion request"
        );
        self.status("Thinking…");
        self.log(
            "send",
            format!(
                "({} msgs) {}",
                history.len(),
                truncate(&prompt_summary, 200)
            ),
        );
        HookAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

// -- fetch_url --------------------------------------------------------------

#[derive(Deserialize, Serialize)]
pub struct FetchUrl;

#[derive(Deserialize)]
pub struct FetchUrlArgs {
    url: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Fetch error: {0}")]
pub struct FetchUrlError(String);

impl From<reqwest::Error> for FetchUrlError {
    fn from(e: reqwest::Error) -> Self {
        FetchUrlError(e.to_string())
    }
}

impl Tool for FetchUrl {
    const NAME: &'static str = "fetch_url";
    type Error = FetchUrlError;
    type Args = FetchUrlArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".into(),
            description: "Fetch a URL and return its text content (HTML). Use this to get the actual content of a web page for accurate rendering.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        info!("fetch_url: {}", args.url);
        let resp = reqwest::get(&args.url).await?.text().await?;
        let truncated = if resp.len() > 20_000 {
            format!("{}... [truncated at 20000 chars]", &resp[..20_000])
        } else {
            resp
        };
        Ok(truncated)
    }
}

// -- update_page ------------------------------------------------------------

#[derive(Clone)]
pub struct UpdatePage {
    pub used: Arc<tokio::sync::Mutex<bool>>,
    pub result: Arc<tokio::sync::Mutex<Option<UpdatePageResult>>>,
    pub openai_client: openai::Client,
    pub image_model: String,
    pub viewport: (u32, u32),
    pub log_tx: broadcast::Sender<LogEntry>,
}

#[derive(Clone)]
pub struct UpdatePageResult {
    pub image_png: Vec<u8>,
    pub url: String,
}

#[derive(Deserialize)]
pub struct UpdatePageArgs {
    url: String,
    page_description: String,
}

impl Tool for UpdatePage {
    const NAME: &'static str = "update_page";
    type Error = UpdatePageError;
    type Args = UpdatePageArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "update_page".into(),
            description: "Render the web page the user sees. Provide the URL for the omnibar and a detailed, accurate visual description of the page body content based on the fetched HTML (NOT browser chrome — just the page itself). This calls an image generator to produce the screenshot. You must call this exactly once per user action.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to show in the browser's omnibar"
                    },
                    "page_description": {
                        "type": "string",
                        "description": "A detailed, accurate visual description of the web page BODY content as it would appear in the browser viewport, based on the actual fetched HTML. Describe the real layout, colors, text, images, buttons, navigation elements, etc. exactly as they appear on the page. Do NOT describe browser chrome (address bar, tabs, window frame) — only the page content itself. Do NOT invent or add content that isn't in the source."
                    }
                },
                "required": ["url", "page_description"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut used = self.used.lock().await;
        if *used {
            return Err(UpdatePageError::AlreadyUsed);
        }
        *used = true;
        drop(used);

        info!(
            "update_page: url={}, description length={}",
            args.url,
            args.page_description.len()
        );
        tracing::debug!(
            description = args.page_description.as_str(),
            "update_page description"
        );

        let (w, h) = pick_image_size(self.viewport);

        let image_model = self.openai_client.image_generation_model(&self.image_model);
        let prompt = format!(
            "Generate a pixel-accurate screenshot of web page content as it would appear inside a browser viewport. \
             Render it exactly as a real web browser would — correct fonts, colors, spacing, and layout. \
             Do NOT include any browser chrome (no address bar, no tabs, no window frame, no OS UI). \
             Only render the page body content itself, edge to edge. \
             Do NOT add any elements, watermarks, annotations, or decorations that are not described below.\n\n\
             The page content to render: {}",
            args.page_description
        );

        let _ = self.log_tx.send(LogEntry {
            kind: "status".into(),
            content: "Generating page image…".into(),
        });

        let response = image_model
            .image_generation_request()
            .prompt(&prompt)
            .width(w)
            .height(h)
            .send()
            .await
            .map_err(|e| UpdatePageError::ImageGen(e.to_string()))?;

        let image_bytes = response.image;

        // Save clean (unannotated) generated image to debug/
        let debug_dir = std::path::Path::new("debug");
        if std::fs::create_dir_all(debug_dir).is_ok() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let filename = format!("page_{ts}.png");
            if let Err(e) = std::fs::write(debug_dir.join(&filename), &image_bytes) {
                warn!("Failed to save debug page image: {e}");
            } else {
                info!("Saved page image to debug/{filename}");
            }
        }

        let mut result = self.result.lock().await;
        *result = Some(UpdatePageResult {
            image_png: image_bytes.clone(),
            url: args.url,
        });

        Ok(format!(
            "Page rendered successfully. Image size: {} bytes",
            image_bytes.len()
        ))
    }
}

fn pick_image_size(viewport: (u32, u32)) -> (u32, u32) {
    let aspect = viewport.0 as f64 / viewport.1 as f64;
    if aspect > 1.25 {
        (1536, 1024)
    } else if aspect < 0.8 {
        (1024, 1536)
    } else {
        (1024, 1024)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UpdatePageError {
    #[error(
        "update_page has already been called for this query. Only one call per user action is allowed."
    )]
    AlreadyUsed,
    #[error("Image generation failed: {0}")]
    ImageGen(String),
}

// ---------------------------------------------------------------------------
// Run agent for a user action
// ---------------------------------------------------------------------------

pub async fn run_agent(
    openai_client: &openai::Client,
    agent_model: &str,
    image_model: &str,
    viewport: (u32, u32),
    log_tx: broadcast::Sender<LogEntry>,
    mut history: Vec<Message>,
    user_message: Message,
) -> Result<(UpdatePageResult, Vec<Message>), AgentRunError> {
    let update_page = UpdatePage {
        used: Arc::new(tokio::sync::Mutex::new(false)),
        result: Arc::new(tokio::sync::Mutex::new(None)),
        openai_client: openai_client.clone(),
        image_model: image_model.to_string(),
        viewport,
        log_tx: log_tx.clone(),
    };
    let result_handle = update_page.result.clone();

    let hook = LogHook { tx: log_tx.clone() };

    let agent = openai_client
        .agent(agent_model)
        .preamble(SYSTEM_PROMPT)
        .tool(FetchUrl)
        .tool(update_page)
        .build();

    let _ = log_tx.send(LogEntry {
        kind: "status".into(),
        content: "Agent started…".into(),
    });

    let response: String = agent
        .prompt(user_message)
        .with_history(&mut history)
        .max_turns(10)
        .with_hook(hook)
        .await
        .map_err(|e| {
            warn!("Agent error: {e}");
            let _ = log_tx.send(LogEntry {
                kind: "error".into(),
                content: format!("Agent error: {e}"),
            });
            AgentRunError::Agent(e.to_string())
        })?;

    if !response.is_empty() {
        tracing::debug!(
            response = truncate(&response, 500).as_str(),
            "Agent final response"
        );
        let _ = log_tx.send(LogEntry {
            kind: "agent".into(),
            content: truncate(&response, 300),
        });
    }

    let result = result_handle.lock().await.take();
    match result {
        Some(r) => {
            let _ = log_tx.send(LogEntry {
                kind: "status".into(),
                content: format!("Done — {}", r.url),
            });
            Ok((r, history))
        }
        None => Err(AgentRunError::NoPageUpdate),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentRunError {
    #[error("Agent error: {0}")]
    Agent(String),
    #[error("Agent did not call update_page")]
    NoPageUpdate,
}

// ---------------------------------------------------------------------------
// Helper: build a user message with an image and text
// ---------------------------------------------------------------------------

pub fn user_message_with_image(text: &str, png_bytes: &[u8]) -> Message {
    let image_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, png_bytes);
    Message::User {
        content: OneOrMany::many(vec![
            UserContent::Text(Text {
                text: text.to_string(),
            }),
            UserContent::Image(Image {
                data: DocumentSourceKind::Base64(image_b64),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                additional_params: None,
            }),
        ])
        .expect("non-empty vec"),
    }
}

// ---------------------------------------------------------------------------
// Draw click circle on image
// ---------------------------------------------------------------------------

pub fn draw_click_circle(
    png_bytes: &[u8],
    x_pct: f64,
    y_pct: f64,
) -> Result<Vec<u8>, image::ImageError> {
    let img = ImageReader::new(std::io::Cursor::new(png_bytes))
        .with_guessed_format()?
        .decode()?;
    let mut rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let cx = (x_pct * w as f64) as i32;
    let cy = (y_pct * h as f64) as i32;
    let radius = 15i32;
    let thickness = 3i32;

    for r in (radius - thickness)..=radius {
        draw_circle_outline(&mut rgba, cx, cy, r, [255, 0, 0, 255]);
    }

    let mut buf = std::io::Cursor::new(Vec::new());
    rgba.write_to(&mut buf, image::ImageFormat::Png)?;
    Ok(buf.into_inner())
}

fn draw_circle_outline(img: &mut image::RgbaImage, cx: i32, cy: i32, r: i32, color: [u8; 4]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut x = r;
    let mut y = 0i32;
    let mut err = 1 - r;

    while x >= y {
        for &(px, py) in &[
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ] {
            if px >= 0 && px < w && py >= 0 && py < h {
                img.put_pixel(px as u32, py as u32, image::Rgba(color));
            }
        }
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}
