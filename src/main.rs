mod agent;
mod routes;
mod session;

use std::sync::Arc;

use clap::Parser;
use rig::client::ProviderClient;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Clone)]
#[command(name = "htmlucinate", about = "A parody web browser rendered by LLMs")]
pub struct Args {
    /// Port to listen on
    #[arg(long, default_value = "3000", env = "PORT")]
    port: u16,

    /// Model to use for the browsing agent
    #[arg(long, default_value = "gpt-5.4", env = "AGENT_MODEL")]
    agent_model: String,

    /// Model to use for image generation
    #[arg(long, default_value = "gpt-image-1.5", env = "IMAGE_MODEL")]
    image_model: String,
}

pub struct AppState {
    pub args: Args,
    pub sessions: dashmap::DashMap<String, session::Session>,
    pub openai_client: rig::providers::openai::Client,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let args = Args::parse();
    let port = args.port;

    let openai_client = rig::providers::openai::Client::from_env();

    let state = Arc::new(AppState {
        args,
        sessions: dashmap::DashMap::new(),
        openai_client,
    });

    let app = routes::router(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");

    tracing::info!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.expect("Server error");
}
