FROM golang:1.24-bookworm

ENV GOPATH=/opt/packages/go \
    GOMODCACHE=/opt/packages/go/pkg/mod \
    GOCACHE=/opt/packages/go/build-cache \
    PATH="/opt/packages/go/bin:$PATH"

WORKDIR /workspace
CMD ["go", "version"]
