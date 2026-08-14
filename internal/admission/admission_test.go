package admission

import (
	"context"
	"errors"
	"testing"
	"time"
)

func testController() *Controller {
	return New(Config{
		LongPromptTokens:       100,
		TotalPromptTokenBudget: 300,
		MaxActiveRequests:      3,
		MaxActiveLongRequests:  1,
		QueueSize:              2,
	})
}

func TestLongRequestsSerialize(t *testing.T) {
	controller := testController()
	releaseFirst, err := controller.Acquire(context.Background(), 150)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = controller.Acquire(ctx, 150)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline, got %v", err)
	}
	releaseFirst()
}

func TestShortRequestBypassesBlockedLong(t *testing.T) {
	controller := testController()
	releaseLong, err := controller.Acquire(context.Background(), 150)
	if err != nil {
		t.Fatal(err)
	}

	longCtx, cancelLong := context.WithCancel(context.Background())
	defer cancelLong()
	longDone := make(chan error, 1)
	go func() {
		_, err := controller.Acquire(longCtx, 150)
		longDone <- err
	}()

	shortCtx, cancelShort := context.WithTimeout(context.Background(), time.Second)
	defer cancelShort()
	releaseShort, err := controller.Acquire(shortCtx, 50)
	if err != nil {
		t.Fatalf("short request should bypass blocked long: %v", err)
	}
	releaseShort()
	cancelLong()
	if err := <-longDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled long request, got %v", err)
	}
	releaseLong()
}

func TestOversizedRequestRunsAlone(t *testing.T) {
	controller := testController()
	release, err := controller.Acquire(context.Background(), 500)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err = controller.Acquire(ctx, 10)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline, got %v", err)
	}
	release()
}

func TestQueueFull(t *testing.T) {
	controller := New(Config{
		LongPromptTokens:       100,
		TotalPromptTokenBudget: 100,
		MaxActiveRequests:      1,
		MaxActiveLongRequests:  1,
		QueueSize:              0,
	})
	release, err := controller.Acquire(context.Background(), 100)
	if err != nil {
		t.Fatal(err)
	}
	_, err = controller.Acquire(context.Background(), 10)
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("expected queue full, got %v", err)
	}
	release()
}
