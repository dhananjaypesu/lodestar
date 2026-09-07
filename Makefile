GO     ?= go
BINDIR ?= bin
PKGS   := ./...

.PHONY: all build test race cover vet fmt bench bench-quick kernels clean

all: vet race build

build:
	@mkdir -p $(BINDIR)
	$(GO) build -o $(BINDIR)/lodestar-bench ./cmd/lodestar-bench

test:
	$(GO) test $(PKGS) -count=1

# Searches run lock-free against the copy-on-write link lists while inserts
# rewrite them, so the race detector is the meaningful test target here, not an
# optional extra.
race:
	$(GO) test $(PKGS) -race -count=1 -timeout 900s

cover:
	$(GO) test $(PKGS) -coverpkg=$(PKGS) -coverprofile=coverage.out -count=1
	$(GO) tool cover -func=coverage.out | tail -1

vet:
	$(GO) vet $(PKGS)

fmt:
	$(GO) fmt $(PKGS)

# The full curve, including the brute-force baseline it is measured against.
bench: build
	./$(BINDIR)/lodestar-bench -n 200000 -dim 128 -queries 1000 -pq 32

bench-quick: build
	./$(BINDIR)/lodestar-bench -n 20000 -dim 64 -queries 300

# The distance kernels, naive against unrolled.
kernels:
	$(GO) test ./internal/vecmath -run xxx -bench . -benchtime 1s

clean:
	rm -rf $(BINDIR) coverage.out *.prof
