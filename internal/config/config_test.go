package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadExpandsEnvironment(t *testing.T) {
	t.Setenv("DEEPSEEK_URL", "http://127.0.0.1:18888")
	path := filepath.Join(t.TempDir(), "config.yaml")
	content := `
server: {listen: ":8888", max_body_bytes: 1024}
deepseek: {base_url: "${DEEPSEEK_URL}", model: "deepseek"}
vision: {enabled: false}
admission:
  long_prompt_tokens: 100
  total_prompt_token_budget: 200
  max_active_requests: 2
  max_active_long_requests: 1
  queue_size: 2
  queue_timeout: 1s
circuit_breaker: {failure_threshold: 3, open_duration: 1s}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DeepSeek.BaseURL != "http://127.0.0.1:18888" {
		t.Fatalf("unexpected URL: %s", cfg.DeepSeek.BaseURL)
	}
}
