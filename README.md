# dgx-llm-simple-proxy

An OpenAI-compatible Go proxy for a DeepSeek text model and a separate vision model. It converts image inputs into cached text descriptions, then sends the transformed request to DeepSeek. It also protects a DGX deployment with token-aware admission control and circuit breakers.

The public model name is `deepseek-v4-flash-0731`.

## Behavior

- `POST /v1/chat/completions` supports streaming and non-streaming requests.
- Request fields are passed through unchanged, including `stream`, `thinking`, `reasoning_effort`, tools, and `parallel_tool_calls`. Only `model` is mapped to the configured DeepSeek model, and image parts are replaced with text descriptions.
- Images supplied as data URLs or remote HTTP(S) URLs are described by the configured vision model.
- Descriptions are cached by SHA-256 image content. Concurrent requests for the same image share one vision call. The cache is in memory and lasts for the process lifetime.
- Prompt tokens are counted with the DeepSeek `/v1/chat/completions/render` endpoint before admission.
- Long requests can be limited independently while short requests continue to use the remaining capacity.
- Three consecutive transport errors or HTTP 5xx responses open the affected upstream circuit for 20 seconds by default. The proxy returns HTTP 503 while open and permits one half-open probe afterward.
- Client cancellation is propagated to both upstreams.

Other endpoints:

- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics` (Prometheus text format)

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

The two-node deployment files under `deploy/dgx` expose the proxy on port 8888 and route DeepSeek traffic to the head node at `192.168.100.10`. DeepSeek must listen on port 18888. Vision is disabled in the DGX deployment because the co-located vision model does not fit reliably alongside the configured 1M-context DeepSeek service.

```sh
cd deploy/dgx
docker compose pull
docker compose up -d
```

## License

MIT
