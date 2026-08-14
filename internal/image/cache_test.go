package image

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
)

func TestDescriptionCacheDeduplicatesConcurrentCompute(t *testing.T) {
	cache := NewDescriptionCache(2)
	var calls atomic.Int32
	start := make(chan struct{})
	compute := func(context.Context) (string, error) {
		calls.Add(1)
		<-start
		return "description", nil
	}

	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			value, err := cache.GetOrCompute(context.Background(), "key", compute)
			if err != nil || value != "description" {
				t.Errorf("unexpected result: value=%q err=%v", value, err)
			}
		}()
	}
	close(start)
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("compute called %d times", calls.Load())
	}
}

func TestDescriptionCacheEvictsLeastRecentlyUsed(t *testing.T) {
	cache := NewDescriptionCache(1)
	compute := func(value string) func(context.Context) (string, error) {
		return func(context.Context) (string, error) { return value, nil }
	}
	if _, err := cache.GetOrCompute(context.Background(), "a", compute("A")); err != nil {
		t.Fatal(err)
	}
	if _, err := cache.GetOrCompute(context.Background(), "b", compute("B")); err != nil {
		t.Fatal(err)
	}
	if cache.Len() != 1 {
		t.Fatalf("unexpected cache size: %d", cache.Len())
	}
}
