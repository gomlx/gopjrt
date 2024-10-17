#!/bin/bash

# This script will install the XlaBuilder C wrapper library and the latest PJRT plugin for gopjrt
# library for Linux (or Windows+WSL). This should be all one need to use it.
#
# Arguments (environment variables):
#
# GOPJRT_INSTALL_DIR: if not empty, defines the directory where to install the library. If empty, it install into `/usr/local`.
# GOPJRT_NOSUDO: if not empty, prevent using sudo to install.
#
# Check install_cuda.sh to additionally install the PJRT plugin for CUDA -- for NVidia GPU support.
#
# To execute this without cloning the repository, one can do:
#
# curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux.sh | bash
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

# Base installation directory:
GOPJRT_INSTALL_DIR="${GOPJRT_INSTALL_DIR:-/usr/local}"
_SUDO="sudo"
if [[ "${GOPJRT_NOSUDO}" != "" ]] ; then
  _SUDO=""
fi

# Fetch address of resources for latest release:
download_urls=$(mktemp --tmpdir gopjrt_urls.XXXXXXXX)
curl -s https://api.github.com/repos/gomlx/gopjrt/releases/latest \
  | grep "browser_download_url" \
  | egrep -wo "https.*gz" \
  > ${download_urls}

# Download XlaBuilder C wrapper library.
url="$(grep gomlx_xlabuilder-linux-amd64.tar.gz "${download_urls}" | head -n 1)"
printf "\nDownloading PJRT CPU plugin from ${url}\n"

if [[ "${_SUDO}" != "" ]] ; then
  echo "Checking sudo authorization for installation"
  ${_SUDO} printf "\tsudo authorized\n"
fi
mkdir -p "${GOPJRT_INSTALL_DIR}"
pushd "${GOPJRT_INSTALL_DIR}"
curl -L "${url}" | ${_SUDO} tar xzv
ls -lh "lib/libgomlx_xlabuilder.a"
popd

# Download PJRT CPU plugin
url="$(grep pjrt_c_api_cpu_plugin.so.gz "${download_urls}" | head -n 1)"
printf "\nDownloading PJRT CPU plugin from ${url}\n"
if [[ "${_SUDO}" != "" ]] ; then
  ${_SUDO} printf "\tsudo authorized\n"
fi
${_SUDO} mkdir -p "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt"
pushd "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt"
curl -L "${url}" | gunzip | ${_SUDO} bash -c 'cat > pjrt_c_api_cpu_plugin.so'
ls -lh pjrt_c_api_cpu_plugin.so
popd

echo "Done."
rm -f "${download_urls}"