use crate::adapters::chat_completions_request::to_chat_completions_payload;
use crate::adapters::chat_completions_response::parse_chat_completions;
use crate::config::normalize_v1_base;
use crate::error::Result;
use crate::model::search::{SearchRequest, SearchResponse};
use crate::providers::http::{build_client, post_json_with_status};
use reqwest::Client;
use std::time::Duration;

#[derive(Clone)]
pub struct OpenAICompatProvider {
    client: Client,
    api_url: String,
    api_key: String,
    model: String,
    include_web_search_tool: bool,
}

impl OpenAICompatProvider {
    pub fn new(
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        include_web_search_tool: bool,
        timeout: Duration,
    ) -> Self {
        Self::with_client(
            build_client(timeout),
            api_url,
            api_key,
            model,
            include_web_search_tool,
        )
    }

    /// Construct with an externally provided `reqwest::Client`. Used by
    /// `SearchService::new` to share one tuned client; the `new(..., timeout)`
    /// form remains for callers that want a per-provider client (tests).
    pub fn with_client(
        client: Client,
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        include_web_search_tool: bool,
    ) -> Self {
        Self {
            client,
            // Mirror the Responses provider: accept root URLs, `/v1` bases, or
            // full endpoints, and converge on a `/v1` base. Without this,
            // `https://api.openai.com` would produce
            // `https://api.openai.com/chat/completions` (missing `/v1`).
            api_url: normalize_v1_base(&api_url.into()),
            api_key: api_key.into(),
            model: model.into(),
            include_web_search_tool,
        }
    }

    pub fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.api_url)
    }

    pub async fn search(&self, request: &SearchRequest) -> Result<SearchResponse> {
        // Honor per-request model overrides (e.g. WebSearchInput.model) the same
        // way the Responses path does; fall back to the provider default only when
        // the request leaves the field empty.
        let model = if request.model.trim().is_empty() {
            self.model.as_str()
        } else {
            request.model.as_str()
        };
        let payload = to_chat_completions_payload(request, model, self.include_web_search_tool);
        let endpoint = self.endpoint();

        // Relays may transiently rate-limit or lose an upstream worker.
        // Retry temporary failures with bounded exponential backoff, while
        // returning authentication and other permanent 4xx errors immediately.
        for attempt in 0..4 {
            match post_json_with_status(
                &self.client,
                &endpoint,
                &self.api_key,
                &payload,
                "OpenAI-compatible",
            )
            .await
            {
                Ok(raw) => return parse_chat_completions(&raw),
                Err(failure)
                    if attempt < 3
                        && (failure.status.is_none()
                            || matches!(failure.status, Some(429 | 500 | 502 | 503 | 504))) =>
                {
                    tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
                }
                Err(failure) => return Err(failure.error),
            }
        }

        unreachable!("retry loop always returns on its final attempt")
    }
}
