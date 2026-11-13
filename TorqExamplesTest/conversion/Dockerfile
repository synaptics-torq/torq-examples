# Host-side model conversion container.
#
# Purpose: run torq-examples' setup_demos.py (model fetch + torq-compile)
# on the x86 CI host. The host runner is Debian 11 (glibc 2.31) and cannot
# run the compiler wheel natively; this image provides a compatible base.
#
# Requirements the base must satisfy (compiler wheel is manylinux_2_39):
#   - glibc >= 2.39
#   - libstdc++ providing GLIBCXX_3.4.32 (GCC 13.x current)
# ubuntu:24.04 satisfies both; test_host_setup_environment proves it at
# runtime (compiler import + torq-compile --version inside the container).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        libstdc++6 ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
