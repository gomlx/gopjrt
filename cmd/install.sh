#!/bin/bash

# This script will install the XlaBuilder C wrapper library and the latest PJRT plugin for gopjrt
# library. This should be all one need to use it.
#
# It uses `sudo` to gain access to admin to create the directories and uncompress the files.
#
# Check install_cuda.sh to additionally install the PJRT plugin for CUDA -- for NVidia GPU support.
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

# Base installation directory:
INSTALL_DIR="${INSTALL_DIR:-/usr/local}"

# Fetch address of resources for latest release:
download_urls=$(mktemp --tmpdir gopjrt_urls.XXXXXXXX)
curl -s https://api.github.com/repos/gomlx/gopjrt/releases/latest \
  | grep "browser_download_url" \
  | egrep -wo "https.*gz" \
  > ${download_urls}

# Download XlaBuilder C wrapper library.
url="$(grep gomlx_xlabuilder-linux-amd64.tar.gz "${download_urls}" | head -n 1)"
printf "\nDownloading PJRT CPU plugin from ${url}\n"
sudo printf "\tsudo authorized\n"
pushd "${INSTALL_DIR}"
curl -L "${url}" | sudo tar xzv
ls -lh "lib/libgomlx_xlabuilder.so"
popd

# Download PJRT CPU plugin
url="$(grep pjrt_c_api_cpu_plugin.so.gz "${download_urls}" | head -n 1)"
printf "\nDownloading PJRT CPU plugin from ${url}\n"
sudo printf "\tsudo authorized\n"
sudo mkdir -p "${INSTALL_DIR}/lib/gomlx/pjrt"
pushd "${INSTALL_DIR}/lib/gomlx/pjrt"
curl -L "${url}" | gunzip | sudo bash -c 'cat > pjrt_c_api_cpu_plugin.so'
ls -lh pjrt_c_api_cpu_plugin.so
popd

echo "Done."
rm -f "${download_urls}"