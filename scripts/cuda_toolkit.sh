#!/usr/bin/env bash
# The task image has the driver's libraries but no CUDA toolkit (no nvcc). Engines that compile CUDA code
# (Cohere megakernel, MPK) need it. Installs the toolkit from NVIDIA's apt repository for this Debian.
set -euo pipefail
# 13.2 is what the driver (595.71) reports and what current torch wheels (cu132) are built for.
CUDA_PKG="${CUDA_PKG:-cuda-toolkit-13-2}"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  . /etc/os-release
  # NVIDIA's repository names the Arm server platform "sbsa" (there is no arm64 directory for Debian).
  case "$(uname -m)" in x86_64) cuda_arch=x86_64 ;; aarch64) cuda_arch=sbsa ;; *) echo "no CUDA apt repository for $(uname -m)" >&2; exit 1 ;; esac
  curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID}/${cuda_arch}/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/cuda-keyring.deb >/dev/null
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$CUDA_PKG" cmake ninja-build libssl-dev pkg-config >/dev/null
fi
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
echo "MK_ENV arch=$(uname -m) nvcc=$(nvcc --version | grep -oE 'release [0-9.]+' ) cmake=$(cmake --version | head -1 | grep -oE '[0-9.]+') ninja=$(ninja --version)"
