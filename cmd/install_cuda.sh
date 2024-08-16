#!/bin/bash

# This script will install the CUDA PJRT plugin, to add support for Nvidia GPUs.
# You should run `install.sh` first to install XlaBuilder C wrapper library (and the CPU plugin).
#
# It uses `sudo` to gain access to admin to create the directories and uncompress the files.
#
# See: https://github.com/gomlx/gopjrt?#installing
set -e

# Base installation directory:
INSTALL_DIR="${INSTALL_DIR:-/usr/local}"

# Create a new virtual environment for Python (if one is not given).
if [[ "${JAX_VENV_DIR}" == "" ]] ; then
  tmp_venv_dir="$(mktemp -d --tmpdir gopjrt_cuda_venv.XXXXXXXX)"
  JAX_VENV_DIR="${tmp_venv_dir}/jax"
  python3 -m venv "${JAX_VENV_DIR}"
fi

# Install jax[cuda12].
source "${JAX_VENV_DIR}/bin/activate"
printf "\nInstalling jax[cuda12] in ${JAX_VENV_DIR}:\n"
pip install "jax[cuda12]"

# Copy over PJRT CUDA plugin:
sudo printf "\t- sudo authorized\n"
sudo mkdir -p "${INSTALL_DIR}/lib/gomlx/pjrt"
sudo rm -f "${INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"
sudo cp -f "${JAX_VENV_DIR}/lib/python3."*"/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so" \
  "${INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"
ls -lh "${INSTALL_DIR}/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so"

# Copy over Nvidia drivers installed for JAX.
printf "\t- removing previous Nvidia drivers installation for gomlx/gopjrt\n"
sudo mkdir -p "${INSTALL_DIR}/lib/gomlx/nvidia"
sudo rm -rf "${INSTALL_DIR}/lib/gomlx/nvidia/"*
printf "\t- copying over Nvidia drivers from Jax installation\n"
sudo cp -r "${JAX_VENV_DIR}/lib/python3."*"/site-packages/nvidia/"* \
  "${INSTALL_DIR}/lib/gomlx/nvidia/"
ls -lh "${INSTALL_DIR}/lib/gomlx/nvidia/"

# Clean up and finish.
if [[ "${tmp_venv_dir}" != "" ]] ; then
  echo "Cleaning up ${tmp_venv_dir}"
  rm -rf "${tmp_venv_dir}"
fi
echo "Done."
