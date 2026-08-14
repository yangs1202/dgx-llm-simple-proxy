package image

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
)

type Reader struct {
	maxBytes    int64
	allowRemote bool
	client      *http.Client
}

type Data struct {
	MediaType string
	Bytes     []byte
}

func NewReader(maxBytes int64, allowRemote, allowPrivate bool, timeoutClient *http.Client) *Reader {
	client := timeoutClient
	if allowRemote && !allowPrivate {
		transport := http.DefaultTransport.(*http.Transport).Clone()
		dialer := &net.Dialer{}
		transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, fmt.Errorf("split image host: %w", err)
			}
			addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
			if err != nil {
				return nil, fmt.Errorf("resolve image host: %w", err)
			}
			for _, address := range addresses {
				if !isPublic(address) {
					return nil, errors.New("private image hosts are not allowed")
				}
			}
			return dialer.DialContext(ctx, network, net.JoinHostPort(addresses[0].String(), port))
		}
		client = &http.Client{
			Timeout:   timeoutClient.Timeout,
			Transport: transport,
			CheckRedirect: func(request *http.Request, via []*http.Request) error {
				if len(via) >= 5 {
					return errors.New("too many image redirects")
				}
				return nil
			},
		}
	}
	return &Reader{maxBytes: maxBytes, allowRemote: allowRemote, client: client}
}

func (r *Reader) Read(ctx context.Context, source string) (Data, error) {
	if strings.HasPrefix(source, "data:") {
		return r.readDataURL(source)
	}
	if !r.allowRemote {
		return Data{}, errors.New("remote images are disabled")
	}
	return r.readRemote(ctx, source)
}

func (r *Reader) readDataURL(source string) (Data, error) {
	header, payload, ok := strings.Cut(source, ",")
	if !ok || !strings.HasPrefix(header, "data:") {
		return Data{}, errors.New("invalid image data URL")
	}
	metadata := strings.TrimPrefix(header, "data:")
	parts := strings.Split(metadata, ";")
	mediaType := parts[0]
	if !strings.HasPrefix(mediaType, "image/") {
		return Data{}, errors.New("data URL is not an image")
	}
	var data []byte
	var err error
	if slicesContain(parts[1:], "base64") {
		data, err = base64.StdEncoding.DecodeString(payload)
	} else {
		var decoded string
		decoded, err = url.PathUnescape(payload)
		data = []byte(decoded)
	}
	if err != nil {
		return Data{}, fmt.Errorf("decode image data URL: %w", err)
	}
	if int64(len(data)) > r.maxBytes {
		return Data{}, errors.New("image exceeds configured size limit")
	}
	return Data{MediaType: mediaType, Bytes: data}, nil
}

func (r *Reader) readRemote(ctx context.Context, source string) (Data, error) {
	parsed, err := url.Parse(source)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return Data{}, errors.New("image URL must use HTTP or HTTPS")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source, nil)
	if err != nil {
		return Data{}, fmt.Errorf("create image request: %w", err)
	}
	request.Header.Set("User-Agent", "dgx-llm-simple-proxy/1.0")
	response, err := r.client.Do(request)
	if err != nil {
		return Data{}, fmt.Errorf("fetch image: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Data{}, fmt.Errorf("fetch image: HTTP %d", response.StatusCode)
	}
	if response.ContentLength > r.maxBytes {
		return Data{}, errors.New("image exceeds configured size limit")
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, r.maxBytes+1))
	if err != nil {
		return Data{}, fmt.Errorf("read image: %w", err)
	}
	if int64(len(data)) > r.maxBytes {
		return Data{}, errors.New("image exceeds configured size limit")
	}
	mediaType, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if !strings.HasPrefix(mediaType, "image/") {
		return Data{}, errors.New("remote URL did not return an image")
	}
	return Data{MediaType: mediaType, Bytes: data}, nil
}

func isPublic(address netip.Addr) bool {
	return address.IsValid() && !address.IsPrivate() && !address.IsLoopback() &&
		!address.IsLinkLocalUnicast() && !address.IsLinkLocalMulticast() &&
		!address.IsMulticast() && !address.IsUnspecified()
}

func slicesContain(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
