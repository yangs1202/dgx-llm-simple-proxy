.PHONY: build test verify

build:
	go build -o bin/dgx-llm-simple-proxy ./cmd/proxy

test:
	go test ./...

verify:
	test -z "$$(gofmt -l $$(find . -name '*.go' -not -path './vendor/*'))"
	go vet ./...
	go test -race ./...
