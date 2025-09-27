#!/bin/bash

# This script will install the CUDA 13 PJRT plugin, to add support for Nvidia GPUs in Linux (or Windows+WSL).
# You should run `install_linux.sh` first to install XlaBuilder C wrapper library (and the CPU plugin).
#
# It requires Nvidia set of drivers > 580 to work.  
#
# Arguments (environment variables):
#
# GOPJRT_INSTALL_DIR: if not empty, defines the directory where to install the library. If empty, it install into `/usr/local`.
# GOPJRT_NOSUDO: if not empty, prevent using sudo to install.
#
# It uses `sudo` to gain access to admin to create the directories and uncompress the files.
#
# To execute this without cloning the repository, one can do:
#
# curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

# Base installation directory:
GOPJRT_INSTALL_DIR="${GOPJRT_INSTALL_DIR:-/usr/local}"
_SUDO="sudo"
if [[ "${GOPJRT_NOSUDO}" != "" ]] ; then
  _SUDO=""
fi

# Create a new virtual environment for Python (if one is not given).
if [[ "${JAX_VENV_DIR}" == "" ]] ; then
  tmp_venv_dir="$(mktemp -d --tmpdir gopjrt_cuda_venv.XXXXXXXX)"
  JAX_VENV_DIR="${tmp_venv_dir}/jax"
  python3 -m venv "${JAX_VENV_DIR}"
fi

# Install jax[cuda13].
source "${JAX_VENV_DIR}/bin/activate"
printf "\nInstalling jax[cuda13] in ${JAX_VENV_DIR}:\n"
pip install "jax[cuda13]"

# Copy over PJRT CUDA plugin:
if [[ "${_SUDO}" != "" ]] ; then
  echo "Checking sudo authorization for installation"
  ${_SUDO} printf "\tsudo authorized\n"
fi
${_SUDO} mkdir -p "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt"
${_SUDO} rm -f "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"
${_SUDO} cp -f "${JAX_VENV_DIR}/lib/python3."*"/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so" \
  "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"
ls -lh "${GOPJRT_INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"

# Copy over Nvidia drivers installed for JAX.
printf "\t- removing previous Nvidia drivers installation for gomlx/gopjrt\n"
${_SUDO} mkdir -p "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia"
${_SUDO} rm -rf "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/"*
printf "\t- copying over Nvidia drivers from Jax installation\n"
${_SUDO} cp -r "${JAX_VENV_DIR}/lib/python3."*"/site-packages/nvidia/"* \
  "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/"
ls -lh "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/"

# Clean up and finish.
if [[ "${tmp_venv_dir}" != "" ]] ; then
  echo "Cleaning up ${tmp_venv_dir}"
  rm -rf "${tmp_venv_dir}"
fi

# Create a bin/ directory and link ptxas binary, so it can be used with
# XLA_FLAGS=--xla_gpu_cuda_data_dir=<path_to_nvidia_installed_files> to point
# PJRT to the ptxas binary to use.
${_SUDO} mkdir -p "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/bin"
${_SUDO} ln -sf "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/cu13/bin/ptxas" \
  "${GOPJRT_INSTALL_DIR}/lib/gomlx/nvidia/bin/ptxas"

echo "Done."
