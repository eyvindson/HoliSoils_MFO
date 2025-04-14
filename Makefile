NIX_OPTIONS ?= --accept-flake-config

all: build

.PHONY: build
build:
	$(RM) build
	nix build $(NIX_OPTIONS)

dist:
	$$(nix build $(NIX_OPTIONS) --json .#image | jq -r .[0].outputs.out) | podman load

shell:
	nix develop $(NIX_OPTIONS)

###

include release-container.mk

nix-%:
	@echo "run inside nix devShell: $*"
	nix develop $(NIX_OPTIONS) --command $(MAKE) $*

.PHONY: FORCE
FORCE:
