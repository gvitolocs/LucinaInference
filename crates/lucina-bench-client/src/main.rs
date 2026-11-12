//! lucina-bench-client: time streaming OpenAI chat completions identically against any server (llama-server, infr
//! serve). Per request: time to first content token, decode tokens/s, token counts, the server's own `timings`.
//!
//!   lucina-bench-client --url http://127.0.0.1:<bench port> --requests requests.jsonl --out results.json
//!       [--api-key-file F] [--model NAME] [--repeat N] [--timeout-s 600]
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

/// What one SSE `data:` line carries.
#[derive(Debug, Default, PartialEq)]
pub struct Chunk {
    pub done: bool,
    pub content: bool,
    pub completion_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub timings: Option<Value>,
}

/// Parse one line of the stream; `None` for lines that are not `data:` events.
pub fn parse_line(line: &str) -> Option<Chunk> {
    let data = line.strip_prefix("data:")?.trim();
    if data == "[DONE]" {
        return Some(Chunk { done: true, ..Default::default() });
    }
    let v: Value = serde_json::from_str(data).ok()?;
    let mut c = Chunk::default();
    if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
        let delta = choice.get("delta");
        let nonempty = |k: &str| delta.and_then(|d| d.get(k)).and_then(|s| s.as_str()).is_some_and(|s| !s.is_empty());
        c.content = nonempty("content") || nonempty("reasoning_content");
        c.finish_reason = choice.get("finish_reason").and_then(|f| f.as_str()).map(String::from);
    }
    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        c.completion_tokens = u.get("completion_tokens").and_then(|x| x.as_u64());
        c.prompt_tokens = u.get("prompt_tokens").and_then(|x| x.as_u64());
    }
    c.timings = v.get("timings").cloned();
    Some(c)
}

/// Decode tokens/s after the first token.
pub fn decode_tps(tokens: u64, ttft_s: f64, total_s: f64) -> Option<f64> {
    let span = total_s - ttft_s;
    (tokens > 1 && span > 0.0).then(|| (tokens - 1) as f64 / span)
}

struct Args {
    url: String,
    requests: String,
    out: String,
    key: Option<String>,
    model: String,
    repeat: usize,
    timeout: u64,
}

fn args() -> Result<Args> {
    let mut a = Args {
        url: String::new(),
        requests: String::new(),
        out: String::new(),
        key: None,
        model: "qwen".into(),
        repeat: 1,
        timeout: 600,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().with_context(|| format!("missing value for {k}"))?;
        match k.as_str() {
            "--url" => a.url = v,
            "--requests" => a.requests = v,
            "--out" => a.out = v,
            "--api-key-file" => a.key = Some(std::fs::read_to_string(&v)?.trim().to_string()),
            "--model" => a.model = v,
            "--repeat" => a.repeat = v.parse()?,
            "--timeout-s" => a.timeout = v.parse()?,
            _ => bail!("unknown argument {k}"),
        }
    }
    if a.url.is_empty() || a.requests.is_empty() || a.out.is_empty() {
        bail!("need --url, --requests and --out");
    }
    Ok(a)
}

fn run_one(client: &reqwest::blocking::Client, a: &Args, req: &Value) -> Result<Value> {
    let mut body = req.clone();
    let obj = body.as_object_mut().context("request line is not an object")?;
    obj.remove("id");
    obj.insert("stream".into(), json!(true));
    obj.insert("model".into(), json!(a.model));
    obj.insert("stream_options".into(), json!({"include_usage": true}));
    let mut rb = client.post(format!("{}/v1/chat/completions", a.url.trim_end_matches('/'))).json(&body);
    if let Some(k) = &a.key {
        rb = rb.bearer_auth(k);
    }
    let t0 = Instant::now();
    let resp = rb.send()?;
    if !resp.status().is_success() {
        bail!("HTTP {}: {}", resp.status(), resp.text().unwrap_or_default());
    }
    let (mut ttft, mut n_events, mut ctoks, mut ptoks, mut finish, mut timings) = (None, 0u64, None, None, None, None);
    for line in BufReader::new(resp).lines() {
        let line = line?;
        let Some(c) = parse_line(&line) else { continue };
        if c.done {
            break;
        }
        if c.content {
            n_events += 1;
            ttft.get_or_insert(t0.elapsed().as_secs_f64());
        }
        ctoks = c.completion_tokens.or(ctoks);
        ptoks = c.prompt_tokens.or(ptoks);
        finish = c.finish_reason.or(finish);
        timings = c.timings.or(timings);
    }
    let total = t0.elapsed().as_secs_f64();
    let ttft = ttft.unwrap_or(total);
    let tps = decode_tps(ctoks.unwrap_or(n_events), ttft, total);
    Ok(json!({"ttft_s": ttft, "total_s": total, "n_events": n_events, "completion_tokens": ctoks,
              "prompt_tokens": ptoks, "decode_tps": tps, "finish_reason": finish, "timings": timings}))
}

fn main() -> Result<()> {
    let a = args()?;
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(a.timeout)).build()?;
    let lines = std::fs::read_to_string(&a.requests)?;
    let mut out = vec![];
    let mut failed = false;
    for line in lines.lines().filter(|l| !l.trim().is_empty()) {
        let req: Value = serde_json::from_str(line)?;
        let id = req.get("id").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        for rep in 0..a.repeat {
            let mut r = match run_one(&client, &a, &req) {
                Ok(v) => v,
                Err(e) => {
                    failed = true;
                    json!({"error": e.to_string()})
                }
            };
            r["id"] = json!(id);
            r["rep"] = json!(rep);
            println!(
                "{id} {rep} ttft {:.3} s decode {} t/s tokens {} finish {}",
                r["ttft_s"].as_f64().unwrap_or(f64::NAN),
                r["decode_tps"].as_f64().map_or("-".into(), |v| format!("{v:.1}")),
                r["completion_tokens"],
                r["finish_reason"].as_str().or(r["error"].as_str()).unwrap_or("-")
            );
            out.push(r);
        }
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&out)?)?;
    if failed {
        bail!("some requests failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_lines() {
        assert_eq!(parse_line("event: x"), None);
        assert!(parse_line("data: [DONE]").unwrap().done);
        let c = parse_line(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#).unwrap();
        assert!(c.content);
        let c = parse_line(r#"data: {"choices":[{"delta":{"content":""}}]}"#).unwrap();
        assert!(!c.content, "empty content is not a token event");
        let c = parse_line(r#"data: {"choices":[{"delta":{"reasoning_content":"r"}}]}"#).unwrap();
        assert!(c.content);
        let c = parse_line(r#"data: {"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":9}}"#).unwrap();
        assert_eq!((c.completion_tokens, c.prompt_tokens), (Some(3), Some(9)));
        let c = parse_line(r#"data: {"choices":[{"delta":{},"finish_reason":"length"}],"timings":{"predicted_per_second":42.0}}"#)
            .unwrap();
        assert_eq!(c.finish_reason.as_deref(), Some("length"));
        assert_eq!(c.timings.unwrap()["predicted_per_second"], 42.0);
    }

    #[test]
    fn decode_rate() {
        assert_eq!(decode_tps(11, 1.0, 2.0), Some(10.0));
        assert_eq!(decode_tps(1, 1.0, 2.0), None);
        assert_eq!(decode_tps(5, 2.0, 2.0), None);
    }
}
