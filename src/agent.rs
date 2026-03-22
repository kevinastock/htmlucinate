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

const SYSTEM_PROMPT: &str = r#"You are the rendering engine of a parody web browser called "HTMLucinate".

When the user types a URL or search query into the omnibar, your job is to:
1. Figure out what web page they are trying to visit. If you need to search, you can use
   fetch_url with the DuckDuckGo API: https://api.duckduckgo.com/?q=YOUR+QUERY&format=json&no_html=1
   This returns instant answers, related topics, and abstracts that can help you understand what
   the user is looking for.
2. Optionally fetch the real page with fetch_url to get inspiration for layout/content.
3. Call update_page with a detailed visual description of what the rendered page looks like.

When the user clicks somewhere on the current page, you'll receive the screenshot with
a red circle indicating where they clicked. Interpret the click (link, button, etc.)
and produce the resulting page via update_page.

When the user asks to scroll down ("page down"), produce the next screenful of content
via update_page.

Important guidelines:
- Always call update_page exactly once per user action.
- The page_description you give to update_page should be a rich, detailed visual description
  of the web page as it would appear in a browser. Describe layout, colors, text content,
  images, buttons, links, etc.
- Be creative and humorous — this is a parody browser. Pages should look plausible at first
  glance but can contain whimsical or absurd details.
- Include realistic-looking UI elements (navigation bars, sidebars, ads, cookie banners, etc.)
"#;

// ---------------------------------------------------------------------------
// Logging hook
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LogHook {
    pub tx: broadcast::Sender<LogEntry>,
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
        let _ = self.tx.send(LogEntry {
            kind: "tool_call".into(),
            content: format!("{tool_name}({args})"),
        });
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
        let truncated = if result.len() > 500 {
            format!("{}... [truncated]", &result[..500])
        } else {
            result.to_string()
        };
        info!(tool = tool_name, result_len = result.len(), "Tool result");
        tracing::debug!(tool = tool_name, result = truncated.as_str(), "Tool result content");
        let _ = self.tx.send(LogEntry {
            kind: "tool_result".into(),
            content: format!("{tool_name} → {truncated}"),
        });
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
                    tool_calls.push(format!("{}()", tc.function.name));
                }
                _ => {}
            }
        }
        if !texts.is_empty() {
            let text = texts.join(" ");
            let text_len = text.len();
            let truncated = if text.len() > 500 {
                format!("{}...", &text[..500])
            } else {
                text
            };
            info!(text_len, "Assistant text response");
            tracing::debug!(text = truncated.as_str(), "Assistant text content");
            let _ = self.tx.send(LogEntry {
                kind: "assistant".into(),
                content: truncated,
            });
        }
        if !tool_calls.is_empty() {
            let calls = tool_calls.join(", ");
            info!(calls = calls.as_str(), "Assistant tool calls");
            let _ = self.tx.send(LogEntry {
                kind: "assistant".into(),
                content: format!("Agent calling: {calls}"),
            });
        }
        if texts.is_empty() && tool_calls.is_empty() {
            info!("Assistant responded with no text or tool calls");
            let _ = self.tx.send(LogEntry {
                kind: "assistant".into(),
                content: "Agent responded (no text content)".into(),
            });
        }
        HookAction::Continue
    }

    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        let prompt_summary = match prompt {
            Message::User { content } => {
                let parts: Vec<String> = content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text(t) => {
                            let s = &t.text;
                            if s.len() > 200 {
                                format!("{}...", &s[..200])
                            } else {
                                s.clone()
                            }
                        }
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
        let _ = self.tx.send(LogEntry {
            kind: "user".into(),
            content: format!(
                "Sending to model (history: {} msgs): {}",
                history.len(),
                prompt_summary
            ),
        });
        HookAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

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
            description: "Fetch a URL and return its text content (HTML). Use this to inspect real web pages for layout inspiration.".into(),
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

#[derive(Clone)]
pub struct UpdatePage {
    pub used: Arc<tokio::sync::Mutex<bool>>,
    pub result: Arc<tokio::sync::Mutex<Option<UpdatePageResult>>>,
    pub openai_client: openai::Client,
    pub image_model: String,
    pub viewport: (u32, u32),
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
            description: "Render the web page the user sees. Provide the URL for the omnibar and a detailed visual description of the page. This calls an image generator to produce the screenshot. You must call this exactly once per user action.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to show in the browser's omnibar"
                    },
                    "page_description": {
                        "type": "string",
                        "description": "A detailed visual description of the entire web page as it would appear rendered in a browser. Include layout, colors, text, images, buttons, navigation elements, etc."
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

        let (w, h) = pick_image_size(self.viewport);

        let image_model = self.openai_client.image_generation_model(&self.image_model);
        let prompt = format!(
            "A screenshot of a web browser's page content area showing: {}",
            args.page_description
        );

        let response = image_model
            .image_generation_request()
            .prompt(&prompt)
            .width(w)
            .height(h)
            .send()
            .await
            .map_err(|e| UpdatePageError::ImageGen(e.to_string()))?;

        let image_bytes = response.image;

        // Save generated image to debug/ for inspection
        let debug_dir = std::path::Path::new("debug");
        if std::fs::create_dir_all(debug_dir).is_ok() {
            let filename = format!(
                "page_{}.png",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            );
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
        kind: "assistant".into(),
        content: "Agent started processing...".into(),
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

    // Log the final text response from the agent
    if !response.is_empty() {
        let truncated = if response.len() > 500 {
            format!("{}...", &response[..500])
        } else {
            response.clone()
        };
        let _ = log_tx.send(LogEntry {
            kind: "assistant".into(),
            content: format!("Final response: {truncated}"),
        });
    }

    let result = result_handle.lock().await.take();
    match result {
        Some(r) => {
            let _ = log_tx.send(LogEntry {
                kind: "assistant".into(),
                content: format!("Page rendered: {}", r.url),
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
