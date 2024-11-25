#!/bin/bash

# This script will install the libraries needed by gopjrt for Darwin/Arm64:
#
# - XlaBuilder C wrapper library (and corresponding .h files)
# - CPU PJRT plugin for gopjrt, the thing that actually does the JIT-compilation.
#
# Arguments (environment variables):
#
# - GOPJRT_INSTALL_DIR: if not empty, defines the directory where to install the library. If empty, it install into `/usr/local`.
#   Notice that ${GOPJRT_INSTALL_DIR}/lib must be set in your LD_LIBRARY_CONF -- `/usr/local/lib` usually is included in the path.
# - GOPJRT_NOSUDO: if not empty, it prevents using sudo to install.
# - GOPJRT_METAL: if not empty, also installs Apple/Metal PJRT plugin, taken from Jax[metal].
#   **VERY EXPERIMENTAL** It requires manual recompilation of `xlabuilder` C libraries to include StableHLO support.
#   It adds support for Apple's GPU, but it's not working very well, misses some operations and dtypes (notably float64),
#   but this allows you to experiment with it.
#   When running your GoMLX programs set GOMLX_BACKEND=xla:metal and the default backend creator will use it.
#
# To execute this without cloning the repository, one can do:
#
# curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin_arm64.sh | bash
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

PLATFORM="darwin_arm64"

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

# Download XlaBuilder C wrapper library and PJRT CPU plugin.
url="$(grep gomlx_xlabuilder_${PLATFORM}.tar.gz "${download_urls}" | head -n 1)"
printf "\nDownloading PJRT CPU plugin from ${url}\n"

mkdir -p "${GOPJRT_INSTALL_DIR}"
pushd "${GOPJRT_INSTALL_DIR}"

tar_file=$(mktemp --tmpdir gopjrt_${PLATFORM}.XXXXXXXX)
curl -L "${url}" > "${tar_file}"

if [[ "${_SUDO}" != "" ]] ; then
  echo "Checking sudo authorization for installation"
  ${_SUDO} printf "\tsudo authorized\n"
fi
sudo tar xvzf "${tar_file}"
rm -f "${tar_file}"

popd
rm -f "${download_urls}"

#
# Download PJRT Metal plugin from the "jax-metal" pip package
#
if [[ "${GOPJRT_METAL}" != "" ]] ; then
  PJRT_NAME="pjrt_c_api_metal_plugin.dylib"
  PJRT_PATH="${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt/${PJRT_NAME}"

  # Create a new virtual environment for Python (if one is not given).
  if [[ "${JAX_VENV_DIR}" == "" ]] ; then
    tmp_venv_dir="$(mktemp -d --tmpdir gopjrt_cuda_venv.XXXXXXXX)"
    JAX_VENV_DIR="${tmp_venv_dir}/jax"
    python3 -m venv "${JAX_VENV_DIR}"
  fi

  # Install jax-metal.
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
  deactivate

  # Clean up and finish.
  if [[ "${tmp_venv_dir}" != "" ]] ; then
    echo "Cleaning up ${tmp_venv_dir}"
    rm -rf "${tmp_venv_dir}"
  fi
fi

echo "Done."

