# HTMLucinate Build Plan

## Phase 1: Project Scaffolding
- [x] `cargo init` with 2024 edition
- [x] Add dependencies to Cargo.toml
- [x] Create justfile (build, test, clippy, fmt, run)
- [x] Create GitHub Actions CI workflow
- [x] Create .env.example showing required env vars

## Phase 2: Server Skeleton
- [x] CLI with clap: `--port`, `--agent-model`, `--image-model`
- [x] Axum app with routes
- [x] AppState with session map (DashMap)
- [x] Session struct: agent message history, current image, omnibar url, log entries

## Phase 3: Frontend (single HTML page)
- [x] Sticky navbar with: back button, omnibar, page down button, agent log button
- [x] Mobile: collapses to omnibar + hamburger
- [x] Body area: displays current image, sized to fill viewport below navbar
- [x] Click handler on image: captures relative position, POSTs to /click
- [x] Omnibar submit: POSTs to /query
- [x] Back button: POSTs to /back
- [x] Page down: POSTs to /pagedown
- [x] Agent log button: opens modal, connects to SSE endpoint, streams log entries
- [x] Loading state: show spinner/overlay while waiting for agent response
- [x] Default "new tab" image: generated programmatically at startup

## Phase 4: Agent & Tools
- [x] Agent setup with rig: configurable model, system prompt
- [x] Tool: `fetch_url` — use reqwest to GET a URL
- [x] Tool: `update_page` — generates image via image model, enforces single-use per query
- [x] On click: draw red circle on current image at click position, send image + message to agent
- [x] On query: send user query text to agent
- [x] On page down: send "scroll down" instruction to agent
- [x] On back: truncate message history to N-1, restore previous image
- [x] PromptHook for logging agent activity to SSE stream

## Phase 5: Image Generation
- [x] Use rig's image generation with gpt-image-1 (via `image` feature)
- [x] Aspect ratio: pick closest standard size based on viewport dimensions
- [x] Return image as base64 PNG

## Phase 6: Session Management & Streaming
- [x] Each page load creates a new session (UUID)
- [x] SSE endpoint streams agent log entries via broadcast channel
- [x] Log entries: tool calls, tool results, agent messages

## Phase 7: Polish
- [ ] Error handling: display errors in the UI
- [ ] Graceful agent timeouts
- [ ] Clean up old sessions periodically

## Architecture Notes

```
src/
  main.rs       - CLI args (clap), AppState, server startup
  session.rs    - Session struct, snapshot/back, log broadcast
  agent.rs      - Tools (FetchUrl, UpdatePage), agent runner, click circle drawing, PromptHook
  routes.rs     - Axum router, HTML page, all API handlers, SSE streaming
  index.html    - Frontend with Alpine.js (included via include_str!)
```

Key design decisions:
- Messages serialized as JSON in session for flexibility (rig Message ↔ serde_json::Value)
- UpdatePage tool uses Arc<Mutex> shared state for single-use enforcement and result extraction
- Default image generated programmatically (gradient + card) — no external assets needed
- Agent model and image model configurable via CLI args and env vars
