package proxy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/yangs1202/dgx-llm-simple-proxy/internal/admission"
	"github.com/yangs1202/dgx-llm-simple-proxy/internal/circuit"
	"github.com/yangs1202/dgx-llm-simple-proxy/internal/config"
	imageutil "github.com/yangs1202/dgx-llm-simple-proxy/internal/image"
)

type Server struct {
	cfg           config.Config
	logger        *slog.Logger
	deepClient    *http.Client
	visionClient  *http.Client
	images        *imageutil.Reader
	cache         *imageutil.DescriptionCache
	admission     *admission.Controller
	deepBreaker   *circuit.Breaker
	visionBreaker *circuit.Breaker
	requests      atomic.Uint64
	errors        atomic.Uint64
	rejected      atomic.Uint64
	cacheHits     atomic.Uint64
	cacheMisses   atomic.Uint64
}

func New(cfg config.Config, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	visionClient := &http.Client{Timeout: cfg.Vision.Timeout}
	return &Server{
		cfg:          cfg,
		logger:       logger,
		deepClient:   &http.Client{Transport: &http.Transport{ResponseHeaderTimeout: cfg.DeepSeek.ResponseHeaderTimeout}},
		visionClient: visionClient,
		images:       imageutil.NewReader(cfg.Vision.MaxImageBytes, cfg.Vision.AllowRemoteImages, cfg.Vision.AllowPrivateImageHosts, visionClient),
		cache:        imageutil.NewDescriptionCache(cfg.Vision.CacheEntries),
		admission: admission.New(admission.Config{
			LongPromptTokens:       cfg.Admission.LongPromptTokens,
			TotalPromptTokenBudget: cfg.Admission.TotalPromptTokenBudget,
			MaxActiveRequests:      cfg.Admission.MaxActiveRequests,
			MaxActiveLongRequests:  cfg.Admission.MaxActiveLongRequests,
			QueueSize:              cfg.Admission.QueueSize,
		}),
		deepBreaker:   circuit.New(cfg.CircuitBreaker.FailureThreshold, cfg.CircuitBreaker.OpenDuration),
		visionBreaker: circuit.New(cfg.CircuitBreaker.FailureThreshold, cfg.CircuitBreaker.OpenDuration),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /readyz", s.ready)
	mux.HandleFunc("GET /metrics", s.metrics)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	return s.withAccessLog(s.withAuth(mux))
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	s.requests.Add(1)
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.Server.MaxBodyBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
		return
	}
	payload["model"] = s.cfg.DeepSeek.Model
	normalizeThinking(payload)
	if err := s.replaceImages(r.Context(), payload); err != nil {
		if errors.Is(err, circuit.ErrOpen) {
			s.writeCircuitError(w, s.visionBreaker, err)
			return
		}
		s.writeError(w, http.StatusBadGateway, "vision_error", err.Error())
		return
	}

	body, err := json.Marshal(payload)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	tokens, err := s.renderTokenCount(r.Context(), body)
	if err != nil {
		if errors.Is(err, circuit.ErrOpen) {
			s.writeCircuitError(w, s.deepBreaker, err)
			return
		}
		s.writeError(w, http.StatusBadGateway, "upstream_error", "prompt tokenization failed: "+err.Error())
		return
	}
	queueCtx, cancel := context.WithTimeout(r.Context(), s.cfg.Admission.QueueTimeout)
	defer cancel()
	release, err := s.admission.Acquire(queueCtx, tokens)
	if err != nil {
		s.rejected.Add(1)
		if errors.Is(err, context.DeadlineExceeded) {
			s.writeError(w, http.StatusServiceUnavailable, "overloaded", "admission queue timeout")
			return
		}
		s.writeError(w, http.StatusServiceUnavailable, "overloaded", err.Error())
		return
	}
	defer release()

	response, err := s.doUpstream(r.Context(), s.deepClient, s.deepBreaker, joinURL(s.cfg.DeepSeek.BaseURL, "/v1/chat/completions"), s.cfg.DeepSeek.APIKey, body)
	if err != nil {
		s.writeCircuitError(w, s.deepBreaker, err)
		return
	}
	defer response.Body.Close()
	s.copyResponse(w, response)
}

func normalizeThinking(payload map[string]any) {
	thinking, ok := payload["thinking"].(bool)
	if !ok {
		return
	}
	kwargs, ok := payload["chat_template_kwargs"].(map[string]any)
	if !ok {
		kwargs = make(map[string]any)
		payload["chat_template_kwargs"] = kwargs
	}
	kwargs["thinking"] = thinking
	if effort, ok := payload["reasoning_effort"].(string); ok {
		kwargs["reasoning_effort"] = effort
	}
}

func (s *Server) replaceImages(ctx context.Context, payload map[string]any) error {
	messages, ok := payload["messages"].([]any)
	if !ok {
		return nil
	}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok {
			continue
		}
		parts, ok := message["content"].([]any)
		if !ok {
			continue
		}
		converted := make([]any, 0, len(parts))
		for _, rawPart := range parts {
			part, ok := rawPart.(map[string]any)
			if !ok || !isImagePart(part) {
				converted = append(converted, rawPart)
				continue
			}
			source, err := imageSource(part)
			if err != nil {
				return err
			}
			description, err := s.describeImage(ctx, source)
			if err != nil {
				return err
			}
			converted = append(converted, map[string]any{
				"type": "text",
				"text": "[Image description]\n" + description,
			})
		}
		message["content"] = converted
	}
	return nil
}

func isImagePart(part map[string]any) bool {
	typeName, _ := part["type"].(string)
	return typeName == "image_url" || typeName == "input_image"
}

func imageSource(part map[string]any) (string, error) {
	for _, key := range []string{"image_url", "input_image"} {
		value, exists := part[key]
		if !exists {
			continue
		}
		switch typed := value.(type) {
		case string:
			return typed, nil
		case map[string]any:
			if source, ok := typed["url"].(string); ok {
				return source, nil
			}
		}
	}
	return "", errors.New("image content is missing a URL")
}

func (s *Server) describeImage(ctx context.Context, source string) (string, error) {
	if !s.cfg.Vision.Enabled {
		return "", errors.New("vision processing is disabled")
	}
	data, err := s.images.Read(ctx, source)
	if err != nil {
		return "", fmt.Errorf("read image: %w", err)
	}
	key := fmt.Sprintf("%x", sha256.Sum256(data.Bytes))
	missesBefore := s.cacheMisses.Load()
	description, err := s.cache.GetOrCompute(ctx, key, func(ctx context.Context) (string, error) {
		s.cacheMisses.Add(1)
		return s.callVision(ctx, data)
	})
	if err == nil && s.cacheMisses.Load() == missesBefore {
		s.cacheHits.Add(1)
	}
	return description, err
}

func (s *Server) callVision(ctx context.Context, data imageutil.Data) (string, error) {
	dataURL := "data:" + data.MediaType + ";base64," + base64.StdEncoding.EncodeToString(data.Bytes)
	payload := map[string]any{
		"model": s.cfg.Vision.Model,
		"messages": []any{
			map[string]any{"role": "system", "content": "Describe the image accurately for another language model. Include all visible text, objects, layout, and details relevant to the user's request. Do not answer the user's request."},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "Analyze this image."},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURL}},
			}},
		},
		"temperature": 0,
		"max_tokens":  s.cfg.Vision.MaxTokens,
		"stream":      false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	response, err := s.doUpstream(ctx, s.visionClient, s.visionBreaker, joinURL(s.cfg.Vision.BaseURL, "/v1/chat/completions"), s.cfg.Vision.APIKey, body)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		return "", fmt.Errorf("vision returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(message)))
	}
	var result struct {
		Choices []struct {
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil || len(result.Choices) == 0 {
		return "", errors.New("vision returned an invalid response")
	}
	description := textContent(result.Choices[0].Message.Content)
	if description == "" {
		return "", errors.New("vision returned an empty description")
	}
	return description, nil
}

func textContent(content any) string {
	if text, ok := content.(string); ok {
		return text
	}
	parts, _ := content.([]any)
	var texts []string
	for _, rawPart := range parts {
		part, _ := rawPart.(map[string]any)
		if text, ok := part["text"].(string); ok {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n")
}

func (s *Server) renderTokenCount(ctx context.Context, body []byte) (int, error) {
	renderCtx, cancel := context.WithTimeout(ctx, s.cfg.DeepSeek.RenderTimeout)
	defer cancel()
	response, err := s.doUpstream(renderCtx, s.deepClient, s.deepBreaker, joinURL(s.cfg.DeepSeek.BaseURL, "/v1/chat/completions/render"), s.cfg.DeepSeek.APIKey, body)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		return 0, fmt.Errorf("render returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(message)))
	}
	var result struct {
		TokenIDs []json.Number `json:"token_ids"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decode render response: %w", err)
	}
	return len(result.TokenIDs), nil
}

func (s *Server) doUpstream(ctx context.Context, client *http.Client, breaker *circuit.Breaker, endpoint, apiKey string, body []byte) (*http.Response, error) {
	if err := breaker.Allow(); err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		breaker.Failure()
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+apiKey)
	}
	response, err := client.Do(request)
	if err != nil {
		breaker.Failure()
		return nil, err
	}
	if response.StatusCode >= 500 {
		breaker.Failure()
	} else {
		breaker.Success()
	}
	return response, nil
}

func (s *Server) copyResponse(w http.ResponseWriter, response *http.Response) {
	for key, values := range response.Header {
		if isHopByHop(key) {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	buffer := make([]byte, 32*1024)
	flusher, _ := w.(http.Flusher)
	for {
		read, err := response.Body.Read(buffer)
		if read > 0 {
			if _, writeErr := w.Write(buffer[:read]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

func isHopByHop(header string) bool {
	switch strings.ToLower(header) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, joinURL(s.cfg.DeepSeek.BaseURL, "/health"), nil)
	if err != nil {
		s.writeError(w, http.StatusServiceUnavailable, "not_ready", err.Error())
		return
	}
	response, err := s.deepClient.Do(request)
	if err != nil || response.StatusCode < 200 || response.StatusCode >= 300 {
		if response != nil {
			response.Body.Close()
		}
		s.writeError(w, http.StatusServiceUnavailable, "not_ready", "DeepSeek upstream is not ready")
		return
	}
	response.Body.Close()
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []any{map[string]any{"id": s.cfg.DeepSeek.Model, "object": "model", "owned_by": "yangs1202"}},
	})
}

func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	a := s.admission.Snapshot()
	d := s.deepBreaker.Snapshot()
	v := s.visionBreaker.Snapshot()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "dgx_proxy_requests_total %d\n", s.requests.Load())
	fmt.Fprintf(w, "dgx_proxy_errors_total %d\n", s.errors.Load())
	fmt.Fprintf(w, "dgx_proxy_rejected_total %d\n", s.rejected.Load())
	fmt.Fprintf(w, "dgx_proxy_image_cache_hits_total %d\n", s.cacheHits.Load())
	fmt.Fprintf(w, "dgx_proxy_image_cache_misses_total %d\n", s.cacheMisses.Load())
	fmt.Fprintf(w, "dgx_proxy_active_requests %d\n", a.ActiveRequests)
	fmt.Fprintf(w, "dgx_proxy_active_long_requests %d\n", a.ActiveLong)
	fmt.Fprintf(w, "dgx_proxy_active_prompt_tokens %d\n", a.ActivePromptTokens)
	fmt.Fprintf(w, "dgx_proxy_queued_requests %d\n", a.QueuedRequests)
	fmt.Fprintf(w, "dgx_proxy_circuit_open{upstream=\"deepseek\"} %d\n", boolNumber(d.State != circuit.StateClosed))
	fmt.Fprintf(w, "dgx_proxy_circuit_open{upstream=\"vision\"} %d\n", boolNumber(v.State != circuit.StateClosed))
}

func boolNumber(value bool) int {
	if value {
		return 1
	}
	return 0
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Server.APIKey != "" && r.Header.Get("Authorization") != "Bearer "+s.cfg.Server.APIKey {
			s.writeError(w, http.StatusUnauthorized, "authentication_error", "invalid API key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) withAccessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		s.logger.Info("request", "method", r.Method, "path", r.URL.Path, "duration_ms", time.Since(started).Milliseconds())
	})
}

func (s *Server) writeCircuitError(w http.ResponseWriter, breaker *circuit.Breaker, err error) {
	if errors.Is(err, circuit.ErrOpen) {
		retry := breaker.Snapshot().RetryAfter
		w.Header().Set("Retry-After", strconv.Itoa(max(1, int(retry.Seconds()))))
		s.writeError(w, http.StatusServiceUnavailable, "circuit_open", "upstream circuit is open")
		return
	}
	s.writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
}

func (s *Server) writeError(w http.ResponseWriter, status int, errorType, message string) {
	s.errors.Add(1)
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": message, "type": errorType, "code": nil}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func joinURL(base, path string) string {
	parsed, _ := url.Parse(base)
	parsed.Path = strings.TrimRight(parsed.Path, "/") + path
	return parsed.String()
}
