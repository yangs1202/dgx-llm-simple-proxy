package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server         ServerConfig              `yaml:"server"`
	DeepSeek       UpstreamConfig            `yaml:"deepseek"`
	Upstreams      map[string]UpstreamConfig `yaml:"upstreams"`
	Routes         map[string]RouteConfig    `yaml:"routes"`
	Vision         VisionConfig              `yaml:"vision"`
	Admission      AdmissionConfig           `yaml:"admission"`
	CircuitBreaker CircuitBreakerConfig      `yaml:"circuit_breaker"`
}

type ServerConfig struct {
	Listen            string        `yaml:"listen"`
	MaxBodyBytes      int64         `yaml:"max_body_bytes"`
	ReadHeaderTimeout time.Duration `yaml:"read_header_timeout"`
	IdleTimeout       time.Duration `yaml:"idle_timeout"`
	ShutdownTimeout   time.Duration `yaml:"shutdown_timeout"`
	APIKey            string        `yaml:"api_key"`
}

type UpstreamConfig struct {
	BaseURL               string        `yaml:"base_url"`
	Model                 string        `yaml:"model"`
	APIKey                string        `yaml:"api_key"`
	TokenizePath          string        `yaml:"tokenize_path"`
	RenderTimeout         time.Duration `yaml:"render_timeout"`
	ResponseHeaderTimeout time.Duration `yaml:"response_header_timeout"`
}

type RouteConfig struct {
	Upstream        string          `yaml:"upstream"`
	Model           string          `yaml:"model"`
	ThinkingAdapter ThinkingAdapter `yaml:"thinking_adapter"`
}

type ThinkingAdapter string

const (
	ThinkingPassthrough ThinkingAdapter = "passthrough"
	ThinkingDeepSeek    ThinkingAdapter = "deepseek"
	ThinkingQwen        ThinkingAdapter = "qwen"
)

type VisionConfig struct {
	Enabled                bool          `yaml:"enabled"`
	BaseURL                string        `yaml:"base_url"`
	Model                  string        `yaml:"model"`
	APIKey                 string        `yaml:"api_key"`
	Timeout                time.Duration `yaml:"timeout"`
	MaxTokens              int           `yaml:"max_tokens"`
	MaxImageBytes          int64         `yaml:"max_image_bytes"`
	AllowRemoteImages      bool          `yaml:"allow_remote_images"`
	AllowPrivateImageHosts bool          `yaml:"allow_private_image_hosts"`
	CacheEntries           int           `yaml:"cache_entries"`
}

type AdmissionConfig struct {
	LongPromptTokens       int           `yaml:"long_prompt_tokens"`
	TotalPromptTokenBudget int           `yaml:"total_prompt_token_budget"`
	MaxActiveRequests      int           `yaml:"max_active_requests"`
	MaxActiveLongRequests  int           `yaml:"max_active_long_requests"`
	QueueSize              int           `yaml:"queue_size"`
	QueueTimeout           time.Duration `yaml:"queue_timeout"`
}

type CircuitBreakerConfig struct {
	FailureThreshold int           `yaml:"failure_threshold"`
	OpenDuration     time.Duration `yaml:"open_duration"`
}

func Load(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	decoderInput := os.ExpandEnv(string(raw))
	if err := yaml.Unmarshal([]byte(decoderInput), &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	var errs []error
	if c.Server.Listen == "" {
		errs = append(errs, errors.New("server.listen is required"))
	}
	if c.Server.MaxBodyBytes <= 0 {
		errs = append(errs, errors.New("server.max_body_bytes must be positive"))
	}
	if err := validateURL("deepseek.base_url", c.DeepSeek.BaseURL); err != nil {
		errs = append(errs, err)
	}
	if c.DeepSeek.Model == "" {
		errs = append(errs, errors.New("deepseek.model is required"))
	}
	upstreams := c.UpstreamConfigs()
	for name, upstream := range c.Upstreams {
		if name == "deepseek" {
			errs = append(errs, errors.New("upstreams.deepseek is reserved; configure the default deepseek upstream at top level"))
			continue
		}
		if err := validateURL("upstreams."+name+".base_url", upstream.BaseURL); err != nil {
			errs = append(errs, err)
		}
		if upstream.Model == "" {
			errs = append(errs, fmt.Errorf("upstreams.%s.model is required", name))
		}
	}
	for alias, route := range c.Routes {
		if alias == "" {
			errs = append(errs, errors.New("routes cannot contain an empty alias"))
		}
		if route.Upstream == "" {
			errs = append(errs, fmt.Errorf("routes.%s.upstream is required", alias))
			continue
		}
		upstream, ok := upstreams[route.Upstream]
		if !ok {
			errs = append(errs, fmt.Errorf("routes.%s references unknown upstream %q", alias, route.Upstream))
			continue
		}
		if route.Model == "" && upstream.Model == "" {
			errs = append(errs, fmt.Errorf("routes.%s.model or upstream model is required", alias))
		}
		if route.ThinkingAdapter != "" && !route.ThinkingAdapter.IsValid() {
			errs = append(errs, fmt.Errorf("routes.%s.thinking_adapter must be one of passthrough, deepseek, qwen", alias))
		}
	}
	if c.Vision.Enabled {
		if err := validateURL("vision.base_url", c.Vision.BaseURL); err != nil {
			errs = append(errs, err)
		}
		if c.Vision.Model == "" {
			errs = append(errs, errors.New("vision.model is required when vision is enabled"))
		}
		if c.Vision.MaxImageBytes <= 0 || c.Vision.CacheEntries <= 0 {
			errs = append(errs, errors.New("vision image and cache limits must be positive"))
		}
	}
	if c.Admission.LongPromptTokens <= 0 || c.Admission.TotalPromptTokenBudget <= 0 {
		errs = append(errs, errors.New("admission token limits must be positive"))
	}
	if c.Admission.MaxActiveRequests <= 0 || c.Admission.MaxActiveLongRequests <= 0 {
		errs = append(errs, errors.New("admission active request limits must be positive"))
	}
	if c.Admission.MaxActiveLongRequests > c.Admission.MaxActiveRequests {
		errs = append(errs, errors.New("admission.max_active_long_requests cannot exceed max_active_requests"))
	}
	if c.Admission.QueueSize < 0 || c.Admission.QueueTimeout <= 0 {
		errs = append(errs, errors.New("admission queue_size must be non-negative and queue_timeout positive"))
	}
	if c.CircuitBreaker.FailureThreshold <= 0 || c.CircuitBreaker.OpenDuration <= 0 {
		errs = append(errs, errors.New("circuit breaker values must be positive"))
	}
	return errors.Join(errs...)
}

func (c Config) UpstreamConfigs() map[string]UpstreamConfig {
	upstreams := make(map[string]UpstreamConfig, len(c.Upstreams)+1)
	upstreams["deepseek"] = c.DeepSeek
	for name, upstream := range c.Upstreams {
		upstreams[name] = upstream
	}
	return upstreams
}

func (a ThinkingAdapter) IsValid() bool {
	switch a {
	case ThinkingPassthrough, ThinkingDeepSeek, ThinkingQwen:
		return true
	default:
		return false
	}
}

func validateURL(name, value string) error {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return fmt.Errorf("%s must be an absolute URL", name)
	}
	return nil
}
