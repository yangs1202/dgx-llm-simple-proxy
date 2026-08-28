# dgx-llm-simple-proxy

An OpenAI-compatible Go proxy for configurable text-model upstreams and a separate vision model. It converts image inputs into cached text descriptions, then sends the transformed request to the selected upstream. It also protects a DGX deployment with token-aware admission control and circuit breakers.

The default public model name is `deepseek-v4-flash-0731`.

## Behavior

- `POST /v1/chat/completions` supports streaming and non-streaming requests.
- Request fields are passed through unchanged, including `stream`, `reasoning_effort`, tools, and `parallel_tool_calls`. `model` selects a configured route and is replaced with that route's upstream model; image parts are replaced with text descriptions.
- Images supplied as data URLs or remote HTTP(S) URLs are described by the configured vision model.
- Descriptions are cached by SHA-256 image content. Concurrent requests for the same image share one vision call. The cache is in memory and lasts for the process lifetime.
- Prompt tokens are counted with each upstream's configured `tokenize_path` before admission. It defaults to DeepSeek's `/v1/chat/completions/render` for backward compatibility.
- Long requests can be limited independently while short requests continue to use the remaining capacity.
- Three consecutive transport errors or HTTP 5xx responses open the affected upstream circuit for 20 seconds by default. The proxy returns HTTP 503 while open and permits one half-open probe afterward.
- Client cancellation is propagated to both upstreams.

Model aliases can route requests to another upstream. Add named upstreams and
map the client-facing model name in `routes`:

```yaml
upstreams:
  qwen:
    base_url: "http://127.0.0.1:8892"
    model: "qwen3.8-27b"
    tokenize_path: "/v1/tokenize"
    render_timeout: 180s
    response_header_timeout: 300s
routes:
  deepseek:
    upstream: qwen
    thinking_adapter: qwen
```

With this configuration, `model: "deepseek"` is sent to Qwen. The `qwen`
thinking adapter converts `thinking` booleans or DeepSeek-style
`thinking.type` values to Qwen's `chat_template_kwargs.enable_thinking`. The
`deepseek` adapter performs the reverse conversion. Other request fields and
custom template options are preserved. If no route matches, requests retain
the legacy behavior and use the default `deepseek` upstream.

Other endpoints:

- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics` (Prometheus text format, including selected vLLM KV-cache and request gauges)

## Configuration

Copy the example and edit the upstream addresses and capacity limits:

```sh
cp config.example.yaml config.yaml
```

Environment variables in YAML values are expanded. For example:

```yaml
server:
  api_key: "${PROXY_API_KEY}"
```

The default admission policy allows up to four active requests, but only one prompt above 98,304 tokens. Active prompt tokens are capped at 262,144. A single request larger than that budget is allowed only when no other request is active. Up to eight requests may wait for 60 seconds; excess work receives HTTP 503.

Remote image hosts resolving to loopback, private, link-local, multicast, or unspecified addresses are rejected by default to prevent SSRF. Set `allow_private_image_hosts: true` only in a trusted network when private image URLs are required.

## Run

Requires Go 1.24 or newer.

```sh
go run ./cmd/proxy -config config.yaml
```

Or build a binary:

```sh
make build
./bin/dgx-llm-simple-proxy -config config.yaml
```

## Container

Images for `linux/amd64` and `linux/arm64` are published by GitHub-hosted runners:

```sh
docker pull ghcr.io/yangs1202/dgx-llm-simple-proxy:latest
docker run --rm -p 8888:8888 \
  -v "$PWD/config.yaml:/etc/dgx-llm-simple-proxy/config.yaml:ro" \
  ghcr.io/yangs1202/dgx-llm-simple-proxy:latest
```

When the upstreams run on the Docker host, use host-reachable addresses in `config.yaml`; container loopback refers to the proxy container itself.

## Example request

```sh
curl http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "parallel_tool_calls": true
  }'
```

## Verify

```sh
make verify
```

CI runs formatting checks, `go vet`, race-enabled tests, and a build on GitHub's public `ubuntu-latest` runners. The Package workflow publishes the multi-architecture image to GitHub Container Registry.

## DGX Spark deployment

The two-node deployment files under `deploy/dgx` expose the proxy on port 8888 and route DeepSeek traffic to the head node at `192.168.100.10`. DeepSeek must listen on port 18888. Vision requests use the external `ollama/minimax-m3` endpoint so the configured 1M-context DeepSeek service does not give up GPU memory.

```sh
cd deploy/dgx
cp .env.example .env
# Set VISION_API_KEY in .env and restrict access with chmod 600 .env.
docker compose pull
docker compose up -d
```

## License

MIT
