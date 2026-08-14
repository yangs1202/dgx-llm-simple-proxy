package admission

import (
	"context"
	"errors"
	"sync"
)

var ErrQueueFull = errors.New("admission queue is full")

type Config struct {
	LongPromptTokens       int
	TotalPromptTokenBudget int
	MaxActiveRequests      int
	MaxActiveLongRequests  int
	QueueSize              int
}

type Snapshot struct {
	ActiveRequests     int
	ActiveLong         int
	ActivePromptTokens int
	QueuedRequests     int
}

type Controller struct {
	mu     sync.Mutex
	cfg    Config
	active Snapshot
	queue  []*waiter
}

type waiter struct {
	tokens  int
	long    bool
	ready   chan struct{}
	granted bool
}

func New(cfg Config) *Controller {
	return &Controller{cfg: cfg}
}

func (c *Controller) Acquire(ctx context.Context, tokens int) (func(), error) {
	if tokens < 0 {
		tokens = 0
	}
	w := &waiter{
		tokens: tokens,
		long:   tokens > c.cfg.LongPromptTokens,
		ready:  make(chan struct{}),
	}

	c.mu.Lock()
	if len(c.queue) == 0 && c.canAdmit(w) {
		c.grant(w)
		c.mu.Unlock()
		return c.releaseFunc(w), nil
	}
	if len(c.queue) >= c.cfg.QueueSize {
		c.mu.Unlock()
		return nil, ErrQueueFull
	}
	c.queue = append(c.queue, w)
	c.dispatch()
	c.mu.Unlock()

	select {
	case <-w.ready:
		return c.releaseFunc(w), nil
	case <-ctx.Done():
		c.mu.Lock()
		if w.granted {
			c.mu.Unlock()
			return c.releaseFunc(w), nil
		}
		c.remove(w)
		c.mu.Unlock()
		return nil, ctx.Err()
	}
}

func (c *Controller) Snapshot() Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	snapshot := c.active
	snapshot.QueuedRequests = len(c.queue)
	return snapshot
}

func (c *Controller) canAdmit(w *waiter) bool {
	if c.active.ActiveRequests >= c.cfg.MaxActiveRequests {
		return false
	}
	if w.long && c.active.ActiveLong >= c.cfg.MaxActiveLongRequests {
		return false
	}
	if w.tokens > c.cfg.TotalPromptTokenBudget {
		return c.active.ActiveRequests == 0
	}
	return c.active.ActivePromptTokens+w.tokens <= c.cfg.TotalPromptTokenBudget
}

func (c *Controller) grant(w *waiter) {
	w.granted = true
	c.active.ActiveRequests++
	c.active.ActivePromptTokens += w.tokens
	if w.long {
		c.active.ActiveLong++
	}
}

func (c *Controller) dispatch() {
	for {
		granted := false
		for index, w := range c.queue {
			if !c.canAdmit(w) {
				continue
			}
			c.queue = append(c.queue[:index], c.queue[index+1:]...)
			c.grant(w)
			close(w.ready)
			granted = true
			break
		}
		if !granted {
			return
		}
	}
}

func (c *Controller) remove(target *waiter) {
	for index, w := range c.queue {
		if w == target {
			c.queue = append(c.queue[:index], c.queue[index+1:]...)
			return
		}
	}
}

func (c *Controller) releaseFunc(w *waiter) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			c.mu.Lock()
			c.active.ActiveRequests--
			c.active.ActivePromptTokens -= w.tokens
			if w.long {
				c.active.ActiveLong--
			}
			c.dispatch()
			c.mu.Unlock()
		})
	}
}
