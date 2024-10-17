#!/bin/bash

# This script will install the XlaBuilder C wrapper library and the latest PJRT plugin for gopjrt for DarwinOS/arm64
# platform. This should be all one need to use it.
#
# Arguments (environment variables):
#
# GOPJRT_INSTALL_DIR: if not empty, defines the directory where to install the library. If empty, it install into `/usr/local`.
# GOPJRT_NOSUDO: if not empty, prevent using sudo to install.
#
# To execute this without cloning the repository, one can do:
#
# curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin.sh | bash
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

# Base installation directory:
GOPJRT_INSTALL_DIR="${GOPJRT_INSTALL_DIR:-/usr/local}"
_SUDO="sudo"
if [[ "${GOPJRT_NOSUDO}" != "" ]] ; then
  _SUDO=""
fi

if false ; then
# Fetch address of resources for latest release:
download_urls=$(mktemp --tmpdir gopjrt_urls.XXXXXXXX)
curl -s https://api.github.com/repos/gomlx/gopjrt/releases/latest \
  | grep "browser_download_url" \
  | egrep -wo "https.*gz" \
  > ${download_urls}

# Download XlaBuilder C wrapper library.
url="$(grep gomlx_xlabuilder-darwin-arm64.tar.gz "${download_urls}" | head -n 1)"
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
rm -f "${download_urls}"

fi

#
# Download PJRT Metal plugin from the "jax-metal" pip package
#
PJRT_NAME="pjrt_c_api_cpu_plugin.dylib"
PJRT_PATH="${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt/${PJRT_NAME}"

# Create a new virtual environment for Python (if one is not given).
if [[ "${JAX_VENV_DIR}" == "" ]] ; then
  tmp_venv_dir="$(mktemp -d --tmpdir gopjrt_cuda_venv.XXXXXXXX)"
  JAX_VENV_DIR="${tmp_venv_dir}/jax"
  python3 -m venv "${JAX_VENV_DIR}"
fi

# Install jax[cuda12].
source "${JAX_VENV_DIR}/bin/activate"
printf "\nInstalling jax-metal in ${JAX_VENV_DIR}:\n"
pip install "jax-metal"

# Copy over PJRT CUDA plugin:
if [[ "${_SUDO}" != "" ]] ; then
  echo "Checking sudo authorization for installation"
  ${_SUDO} printf "\tsudo authorized\n"
fi
${_SUDO} mkdir -p "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt"
${_SUDO} rm -f "${PJRT_PATH}"
${_SUDO} cp -f "${JAX_VENV_DIR}/lib/python3."*"/site-packages/jax_plugins/metal_plugin/pjrt_plugin_metal_14.dylib" "${PJRT_PATH}"
ls -lh "${PJRT_PATH}"

# Clean up and finish.
if [[ "${tmp_venv_dir}" != "" ]] ; then
  echo "Cleaning up ${tmp_venv_dir}"
  rm -rf "${tmp_venv_dir}"
fi
echo "Done."




