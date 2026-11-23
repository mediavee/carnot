//! Interactive `AskUser` tool module (`defer_loading: true`, 0 static prefix tokens).
//!
//! Allows the agent to ask structured clarifying questions (single/multi-choice options or free
//! text) whenever a host UI (ACP, RPC, HTTP, REPL, or custom Rust embedder) attaches a
//! [`UserQuestionBroker`]. Returns a clear non-blocking guidance message if no interactive human
//! is connected (e.g. headless CI).

use crate::{Module, UserQuestionBroker};
use carnot_core::{
    CarnotError, Result, ToolEffects, ToolOutput, ToolSpec, UserQuestion, UserQuestionAnswer,
    UserQuestionOption,
};
use carnot_tools::{BoxFuture, Tool};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Shared handle to the active interactive `UserQuestionBroker` (swappable at runtime by ACP/RPC/HTTP/REPL).
pub type SharedQuestionBroker = Arc<Mutex<Option<Arc<dyn UserQuestionBroker>>>>;

/// Module registering the deferred `AskUser` tool (`defer_loading: true`).
#[derive(Debug, Clone)]
pub struct AskUserModule {
    tool: Arc<AskUserTool>,
}

impl AskUserModule {
    pub fn new(broker: SharedQuestionBroker) -> Self {
        Self {
            tool: Arc::new(AskUserTool {
                broker,
                next_req_id: AtomicU64::new(1),
            }),
        }
    }
}

impl Module for AskUserModule {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::clone(&self.tool) as Arc<dyn Tool>]
    }
}

#[derive(Debug)]
struct AskUserTool {
    broker: SharedQuestionBroker,
    next_req_id: AtomicU64,
}

fn parse_questions_input(input: &Value) -> std::result::Result<Vec<UserQuestion>, String> {
    if let Some(arr) = input.get("questions").and_then(Value::as_array) {
        if arr.is_empty() {
            return Err("AskUser `questions` array must not be empty".into());
        }
        let mut out = Vec::with_capacity(arr.len());
        for (idx, q) in arr.iter().take(6).enumerate() {
            let text = q
                .get("question")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("questions[{idx}] missing non-empty `question`"))?;
            let id = q
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("q{}", idx + 1));
            let header = q
                .get("header")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let multi_select = q
                .get("multi_select")
                .or_else(|| q.get("is_multi_select"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let options: Vec<UserQuestionOption> = q
                .get("options")
                .and_then(Value::as_array)
                .map(|opts| {
                    opts.iter()
                        .filter_map(|o| {
                            if let Some(s) = o.as_str() {
                                let clean = s.trim();
                                (!clean.is_empty()).then(|| UserQuestionOption {
                                    label: clean.to_string(),
                                    description: None,
                                })
                            } else {
                                let label = o.get("label").and_then(Value::as_str)?.trim();
                                if label.is_empty() {
                                    return None;
                                }
                                let description = o
                                    .get("description")
                                    .and_then(Value::as_str)
                                    .map(str::trim)
                                    .filter(|s| !s.is_empty())
                                    .map(str::to_string);
                                Some(UserQuestionOption {
                                    label: label.to_string(),
                                    description,
                                })
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.push(UserQuestion {
                id,
                header,
                question: text.to_string(),
                options,
                multi_select,
            });
        }
        return Ok(out);
    }

    // Convenience shorthand: `{ "question": "...", "options": ["A", "B"] }`
    if let Some(text) = input
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let wrapped = json!({ "questions": [input] });
        let mut parsed = parse_questions_input(&wrapped)?;
        if let Some(first) = parsed.first_mut() {
            first.question = text.to_string();
        }
        return Ok(parsed);
    }

    Err("AskUser requires `question` (string) or `questions` (array of question objects)".into())
}

fn format_answers(questions: &[UserQuestion], answers: &[UserQuestionAnswer]) -> String {
    let mut out = String::from("[user-answers]\n");
    for q in questions {
        let ans = answers.iter().find(|a| a.id == q.id);
        let value = match ans {
            Some(a) => {
                let mut parts = Vec::new();
                if !a.selected.is_empty() {
                    parts.push(a.selected.join(", "));
                }
                if let Some(ref txt) = a.text
                    && !txt.trim().is_empty()
                {
                    parts.push(txt.trim().to_string());
                }
                if parts.is_empty() {
                    "(no answer provided)".to_string()
                } else {
                    parts.join(" — ")
                }
            }
            None => "(unanswered)".to_string(),
        };
        out.push_str(&format!("- `{}` ({}): {}\n", q.id, q.question, value));
    }
    out.trim_end().to_string()
}

impl Tool for AskUserTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "AskUser".into(),
            description: "Ask the human user one or more concise clarifying questions (optional multiple-choice `options` or free text) when a missing requirement would materially change the solution.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "Single question shorthand" },
                    "options": {
                        "type": "array",
                        "description": "Optional choices for single-question shorthand (strings or {label, description})",
                        "items": {}
                    },
                    "multi_select": { "type": "boolean", "description": "Allow selecting multiple options" },
                    "questions": {
                        "type": "array",
                        "description": "Structured list of up to 6 clarifying questions",
                        "items": {
                            "type": "object",
                            "required": ["question"],
                            "properties": {
                                "id": { "type": "string" },
                                "header": { "type": "string" },
                                "question": { "type": "string" },
                                "multi_select": { "type": "boolean" },
                                "options": { "type": "array", "items": {} }
                            }
                        }
                    }
                }
            }),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let questions = match parse_questions_input(&input) {
                Ok(q) => q,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };

            let broker = self.broker.lock().await.clone();
            let Some(broker) = broker else {
                return Ok(ToolOutput::err_text(
                    "No interactive user is connected in this session; proceed with your best reasonable assumption.",
                ));
            };

            let req_id = format!("ask-{}", self.next_req_id.fetch_add(1, Ordering::Relaxed));
            let answers = tokio::select! {
                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                res = broker.ask(&req_id, &questions) => match res {
                    Ok(ans) => ans,
                    Err(e) => return Ok(ToolOutput::err_text(format!("AskUser failed: {e}"))),
                }
            };

            let formatted = format_answers(&questions, &answers);
            Ok(ToolOutput::ok_text(formatted).with_details(json!({
                "request_id": req_id,
                "answers": answers,
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockQuestionBroker;

    impl UserQuestionBroker for MockQuestionBroker {
        fn ask<'a>(
            &'a self,
            _request_id: &'a str,
            questions: &'a [UserQuestion],
        ) -> crate::BoxFuture<'a, Result<Vec<UserQuestionAnswer>>> {
            Box::pin(async move {
                Ok(questions
                    .iter()
                    .map(|q| UserQuestionAnswer {
                        id: q.id.clone(),
                        selected: q
                            .options
                            .first()
                            .map(|o| vec![o.label.clone()])
                            .unwrap_or_default(),
                        text: Some("use sqlite".into()),
                    })
                    .collect())
            })
        }
    }

    #[tokio::test]
    async fn ask_user_tool_supports_shorthand_and_structured_questions() {
        let shared: SharedQuestionBroker = Arc::new(Mutex::new(Some(Arc::new(MockQuestionBroker))));
        let module = AskUserModule::new(shared);
        let tool = &module.tools()[0];
        assert!(tool.spec().defer_loading);

        let out = tool
            .execute(
                json!({
                    "question": "Which database?",
                    "options": ["SQLite", "Postgres"]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(!out.is_error);
        let txt = out.text_only();
        assert!(txt.contains("SQLite"), "{txt}");
        assert!(txt.contains("use sqlite"), "{txt}");
    }
}
