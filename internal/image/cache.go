package image

import (
	"container/list"
	"context"
	"sync"
)

type cacheEntry struct {
	key   string
	value string
}

type flight struct {
	done  chan struct{}
	value string
	err   error
}

type DescriptionCache struct {
	mu       sync.Mutex
	capacity int
	items    map[string]*list.Element
	order    *list.List
	flights  map[string]*flight
}

func NewDescriptionCache(capacity int) *DescriptionCache {
	return &DescriptionCache{
		capacity: capacity,
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
		flights:  make(map[string]*flight),
	}
}

func (c *DescriptionCache) GetOrCompute(
	ctx context.Context,
	key string,
	compute func(context.Context) (string, error),
) (string, error) {
	c.mu.Lock()
	if element, ok := c.items[key]; ok {
		c.order.MoveToFront(element)
		value := element.Value.(cacheEntry).value
		c.mu.Unlock()
		return value, nil
	}
	if existing, ok := c.flights[key]; ok {
		c.mu.Unlock()
		select {
		case <-existing.done:
			return existing.value, existing.err
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	current := &flight{done: make(chan struct{})}
	c.flights[key] = current
	c.mu.Unlock()

	value, err := compute(ctx)

	c.mu.Lock()
	current.value = value
	current.err = err
	delete(c.flights, key)
	if err == nil {
		element := c.order.PushFront(cacheEntry{key: key, value: value})
		c.items[key] = element
		if c.order.Len() > c.capacity {
			last := c.order.Back()
			delete(c.items, last.Value.(cacheEntry).key)
			c.order.Remove(last)
		}
	}
	close(current.done)
	c.mu.Unlock()
	return value, err
}

func (c *DescriptionCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}
