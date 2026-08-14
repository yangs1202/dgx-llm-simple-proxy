package proxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yangs1202/dgx-llm-simple-proxy/internal/config"
)

func TestPassthroughAndStreaming(t *testing.T) {
	var completionPayload map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions/render":
			writeJSON(w, http.StatusOK, map[string]any{"token_ids": []int{1, 2, 3}})
		case "/v1/chat/completions":
			if err := json.NewDecoder(r.Body).Decode(&completionPayload); err != nil {
				t.Fatal(err)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "data: first\n\ndata: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	server := httptest.NewServer(New(testConfig(upstream.URL, ""), discardLogger()).Handler())
	defer server.Close()
	body := `{"model":"client-model","messages":[{"role":"user","content":"hello"}],"stream":true,"thinking":{"type":"disabled"},"reasoning_effort":"low","parallel_tool_calls":true,"custom_parameter":{"keep":42}}`
	response, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	gotBody, _ := io.ReadAll(response.Body)

	if response.StatusCode != http.StatusOK || !strings.Contains(string(gotBody), "data: first") {
		t.Fatalf("unexpected response: status=%d body=%s", response.StatusCode, gotBody)
	}
	if completionPayload["model"] != "deepseek-v4-flash-0731" {
		t.Fatalf("model was not mapped: %#v", completionPayload["model"])
	}
	if completionPayload["stream"] != true || completionPayload["parallel_tool_calls"] != true {
		t.Fatalf("stream/tool settings were not passed through: %#v", completionPayload)
	}
	if _, ok := completionPayload["thinking"]; !ok {
		t.Fatal("thinking was dropped")
	}
	if completionPayload["reasoning_effort"] != "low" {
		t.Fatalf("reasoning_effort was not passed through: %#v", completionPayload)
	}
	if _, ok := completionPayload["custom_parameter"]; !ok {
		t.Fatal("unknown parameter was dropped")
	}
}

func TestImageIsDescribedOnceAndReplaced(t *testing.T) {
	var visionCalls atomic.Int32
	vision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		visionCalls.Add(1)
		writeJSON(w, http.StatusOK, map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": "a red square"}}},
		})
	}))
	defer vision.Close()

	var sawDescription atomic.Bool
	deep := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(payload["messages"])
		if strings.Contains(string(encoded), "a red square") && !strings.Contains(string(encoded), "image_url") {
			sawDescription.Store(true)
		}
		if strings.HasSuffix(r.URL.Path, "/render") {
			writeJSON(w, http.StatusOK, map[string]any{"token_ids": []int{1}})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"choices": []any{}})
	}))
	defer deep.Close()

	cfg := testConfig(deep.URL, vision.URL)
	server := httptest.NewServer(New(cfg, discardLogger()).Handler())
	defer server.Close()
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]}]}`
	for range 2 {
		response, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("unexpected status: %d", response.StatusCode)
		}
	}
	if visionCalls.Load() != 1 {
		t.Fatalf("vision calls = %d, want 1", visionCalls.Load())
	}
	if !sawDescription.Load() {
		t.Fatal("image description was not sent to DeepSeek")
	}
}

func TestMetricsIncludesSelectedVLLMMetrics(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, `# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="deepseek"} 0.42
vllm:num_requests_running{model_name="deepseek"} 2
vllm:num_requests_waiting{model_name="deepseek"} 1
vllm:prompt_tokens_total{model_name="deepseek"} 999
`)
	}))
	defer upstream.Close()

	server := httptest.NewServer(New(testConfig(upstream.URL, ""), discardLogger()).Handler())
	defer server.Close()
	response, err := http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)

	for _, metric := range []string{"vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"} {
		if !strings.Contains(string(body), metric) {
			t.Fatalf("metric %s is missing from response:\n%s", metric, body)
		}
	}
	if strings.Contains(string(body), "vllm:prompt_tokens_total") {
		t.Fatalf("unexpected vLLM metric was exposed:\n%s", body)
	}
}

func TestMetricsSurvivesVLLMFailure(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer upstream.Close()

	server := httptest.NewServer(New(testConfig(upstream.URL, ""), discardLogger()).Handler())
	defer server.Close()
	response, err := http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusOK || !strings.Contains(string(body), "dgx_proxy_requests_total") {
		t.Fatalf("unexpected response: status=%d body=%s", response.StatusCode, body)
	}
}

func testConfig(deepURL, visionURL string) config.Config {
	return config.Config{
		Server: config.ServerConfig{MaxBodyBytes: 1 << 20},
		DeepSeek: config.UpstreamConfig{
			BaseURL: deepURL, Model: "deepseek-v4-flash-0731",
			RenderTimeout: time.Second, ResponseHeaderTimeout: time.Second,
		},
		Vision: config.VisionConfig{
			Enabled: visionURL != "", BaseURL: visionURL, Model: "vision",
			Timeout: time.Second, MaxTokens: 100, MaxImageBytes: 1 << 20, CacheEntries: 10,
		},
		Admission: config.AdmissionConfig{
			LongPromptTokens: 100, TotalPromptTokenBudget: 1000,
			MaxActiveRequests: 2, MaxActiveLongRequests: 1, QueueSize: 2, QueueTimeout: time.Second,
		},
		CircuitBreaker: config.CircuitBreakerConfig{FailureThreshold: 2, OpenDuration: time.Second},
	}
}

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
