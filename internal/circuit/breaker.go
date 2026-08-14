package circuit

import (
	"errors"
	"sync"
	"time"
)

var ErrOpen = errors.New("circuit is open")

type State string

const (
	StateClosed   State = "closed"
	StateOpen     State = "open"
	StateHalfOpen State = "half_open"
)

type Snapshot struct {
	State               State
	ConsecutiveFailures int
	RetryAfter          time.Duration
}

type Breaker struct {
	mu               sync.Mutex
	failureThreshold int
	openDuration     time.Duration
	now              func() time.Time
	state            State
	failures         int
	openedAt         time.Time
	probeInFlight    bool
}

func New(failureThreshold int, openDuration time.Duration) *Breaker {
	return &Breaker{
		failureThreshold: failureThreshold,
		openDuration:     openDuration,
		now:              time.Now,
		state:            StateClosed,
	}
}

func (b *Breaker) Allow() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.state == StateClosed {
		return nil
	}
	if b.state == StateOpen && b.now().Sub(b.openedAt) < b.openDuration {
		return ErrOpen
	}
	if b.probeInFlight {
		return ErrOpen
	}
	b.state = StateHalfOpen
	b.probeInFlight = true
	return nil
}

func (b *Breaker) Success() {
	b.mu.Lock()
	b.state = StateClosed
	b.failures = 0
	b.openedAt = time.Time{}
	b.probeInFlight = false
	b.mu.Unlock()
}

func (b *Breaker) Failure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probeInFlight = false
	b.failures++
	if b.failures >= b.failureThreshold {
		b.state = StateOpen
		b.openedAt = b.now()
	}
}

func (b *Breaker) Snapshot() Snapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	retryAfter := time.Duration(0)
	if b.state == StateOpen {
		retryAfter = b.openDuration - b.now().Sub(b.openedAt)
		if retryAfter < 0 {
			retryAfter = 0
		}
	}
	return Snapshot{
		State:               b.state,
		ConsecutiveFailures: b.failures,
		RetryAfter:          retryAfter,
	}
}
