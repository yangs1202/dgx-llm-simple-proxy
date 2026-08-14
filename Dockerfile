# syntax=docker/dockerfile:1.7
FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS build
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w" -o /out/dgx-llm-simple-proxy ./cmd/proxy

FROM gcr.io/distroless/static-debian12:nonroot
LABEL org.opencontainers.image.source="https://github.com/yangs1202/dgx-llm-simple-proxy"
COPY --from=build /out/dgx-llm-simple-proxy /usr/local/bin/dgx-llm-simple-proxy
COPY config.example.yaml /etc/dgx-llm-simple-proxy/config.yaml
USER nonroot:nonroot
EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/dgx-llm-simple-proxy"]
CMD ["-config", "/etc/dgx-llm-simple-proxy/config.yaml"]
