# this file has been modified to allow manual builds.
# normally this would all be done in CI,
# but we don't have access to CI on gitlab.jyu.fi at the moment.
# you can find a version that works on CI in the git history if you need it.

# Required inputs:
CONTAINER_PATH = service/multioptforest
CONTAINER_VERSION = 0.4.0

CONTAINER_DOMAIN := container.repo.kopla.jyu.fi
CONTAINER_RT_REPO_STAGING := kopla-docker-staging
CONTAINER_RT_REPO_PROMOTED := kopla-docker-stable

CONTAINER_NAME = $(CONTAINER_DOMAIN)/$(CONTAINER_PATH)
CONTAINER_TAG = $(CONTAINER_VERSION)

CONTAINER_RT_BUILD_NAME := multiforest
CONTAINER_RT_BUILD_NUMBER := manual-$(CONTAINER_TAG)

# Optional: fill in some meta for local builds
CI_COMMIT_TAG ?= $(shell git describe --exact-match --tags 2>/dev/null)
CI_COMMIT_SHA ?= $(shell git rev-parse HEAD)

# Bash is needed for pipefail
SHELL := bash

container-show:
	# CONTAINER_NAME = $(CONTAINER_NAME)
	# CONTAINER_TAG = $(CONTAINER_TAG)
	#
	# CONTAINER_RT_BUILD_NAME = $(CONTAINER_RT_BUILD_NAME)
	# CONTAINER_RT_BUILD_NUMBER = $(CONTAINER_RT_BUILD_NUMBER)
	#
	# CI_COMMIT_REF_NAME = $(CI_COMMIT_REF_NAME)
	# CI_COMMIT_TAG = $(CI_COMMIT_TAG)

container-show-image:
	@echo $(CONTAINER_NAME):$(CONTAINER_TAG)

container-dist:
	@set -o pipefail
	@echo "# Build the image"
	nix build --json .#image|jq -r .[0].outputs.out|sh|podman load -q
	podman tag localhost/$(CONTAINER_PATH):latest $(CONTAINER_NAME):$(CONTAINER_TAG)
	@echo "# Tag every build as latest"
	podman tag $(CONTAINER_NAME):$(CONTAINER_TAG) $(CONTAINER_NAME):latest

container-clean:
	rm -fr result-image.json

container-publish:
	jf rt build-add-git \
		$(CONTAINER_RT_BUILD_NAME) \
		$(CONTAINER_RT_BUILD_NUMBER)

	jf rt podman-push \
		--build-name "$(CONTAINER_RT_BUILD_NAME)" \
		--build-number "$(CONTAINER_RT_BUILD_NUMBER)" \
		$(CONTAINER_NAME):$(CONTAINER_TAG) \
		$(CONTAINER_RT_REPO_STAGING)

	# Rolling tag cannot be promoted if there is already newer
	# version, so "latest" it's not tagged into the build
	jf rt podman-push \
		$(CONTAINER_NAME):latest \
		$(CONTAINER_RT_REPO_STAGING)

	jf rt build-publish \
		$(CONTAINER_RT_BUILD_NAME) \
		$(CONTAINER_RT_BUILD_NUMBER)

container-promote:
	jf rt build-promote \
		$(CONTAINER_RT_BUILD_NAME) \
		$(CONTAINER_RT_BUILD_NUMBER) \
		$(CONTAINER_RT_REPO_PROMOTED)
