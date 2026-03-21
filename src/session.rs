use std::sync::Arc;

use tokio::sync::broadcast;

/// A log entry from the agent session, streamed to the frontend via SSE.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LogEntry {
    pub kind: String, // "tool_call", "tool_result", "assistant", "error"
    pub content: String,
}

/// A snapshot of the session state at a given point, used for back-button navigation.
#[derive(Clone)]
pub struct Snapshot {
    pub image_png: Arc<Vec<u8>>,
    pub omnibar_url: String,
    /// Number of messages in the agent history at the time of this snapshot.
    pub message_count: usize,
}

/// Represents one browsing session.
pub struct Session {
    /// The current page image (PNG bytes).
    pub current_image: Arc<Vec<u8>>,
    /// The current URL shown in the omnibar.
    pub omnibar_url: String,
    /// Chat messages for the agent (stored as serde_json::Value for flexibility).
    pub messages: Vec<serde_json::Value>,
    /// Stack of snapshots for back-button navigation.
    pub history: Vec<Snapshot>,
    /// Broadcast channel for streaming log entries to the frontend.
    pub log_tx: broadcast::Sender<LogEntry>,
    /// The viewport dimensions last reported by the frontend.
    pub viewport: (u32, u32),
}

impl Session {
    pub fn new(default_image: Arc<Vec<u8>>) -> Self {
        let (log_tx, _) = broadcast::channel(256);
        Session {
            current_image: default_image,
            omnibar_url: String::new(),
            messages: Vec::new(),
            history: Vec::new(),
            log_tx,
            viewport: (1024, 768),
        }
    }

    /// Push a snapshot before processing a new user action.
    pub fn push_snapshot(&mut self) {
        self.history.push(Snapshot {
            image_png: self.current_image.clone(),
            omnibar_url: self.omnibar_url.clone(),
            message_count: self.messages.len(),
        });
    }

    /// Pop to the previous snapshot (back button).
    pub fn go_back(&mut self) -> bool {
        if let Some(snap) = self.history.pop() {
            self.current_image = snap.image_png;
            self.omnibar_url = snap.omnibar_url;
            self.messages.truncate(snap.message_count);
            true
        } else {
            false
        }
    }

    pub fn log(&self, kind: &str, content: &str) {
        let _ = self.log_tx.send(LogEntry {
            kind: kind.to_string(),
            content: content.to_string(),
        });
    }
}
