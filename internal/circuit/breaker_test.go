package circuit

import (
	"errors"
	"testing"
	"time"
)

func TestBreakerOpensAndRecovers(t *testing.T) {
	breaker := New(2, 10*time.Second)
	now := time.Unix(100, 0)
	breaker.now = func() time.Time { return now }

	breaker.Failure()
	if err := breaker.Allow(); err != nil {
		t.Fatalf("closed circuit rejected request: %v", err)
	}
	breaker.Failure()
	if err := breaker.Allow(); !errors.Is(err, ErrOpen) {
		t.Fatalf("expected open circuit, got %v", err)
	}

	now = now.Add(11 * time.Second)
	if err := breaker.Allow(); err != nil {
		t.Fatalf("half-open probe rejected: %v", err)
	}
	if err := breaker.Allow(); !errors.Is(err, ErrOpen) {
		t.Fatalf("second half-open probe should be rejected, got %v", err)
	}
	breaker.Success()
	if err := breaker.Allow(); err != nil {
		t.Fatalf("recovered circuit rejected request: %v", err)
	}
}
