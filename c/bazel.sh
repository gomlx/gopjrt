#!/bin/bash
#
# The following environment variables and flags can be defined:
#
# * STARTUP_FLAGS and BUILD_FLAGS: passed as `bazel ${STARTUP_FLAGS} build <build_target> ${BUILD_FLAGS}.
# * --debug: Compile in debug mode, with symbols for gdb.
# * --output <dir>: Directory passed to `bazel --output_base`. Unfortunately, not sure why, bazel still outputs things
#   to $TEST_TMPDIR and /.cache.
# * <build_target>: Default is ":gomlx_xlabuilder".

BUILD_TARGET=":gomlx_xlabuilder"

export USE_BAZEL_VERSION=7.3.1  # Latest as of this writing.

# Versions 8 and above don't work. They seem to require blzmod (and the compatibility --enable_workspace build option
# doesn't seem to work the same):
# export USE_BAZEL_VERSION=last_green
# export USE_BAZEL_VERSION=8.0.0-pre.20240911.1

DEBUG=0
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG=1
      shift
      ;;
    --output)
      shift
      echo "Output directory set to $1"
      OUTPUT_DIR="--output_base=$1"
      shift
      ;;
    -*|--*)
      echo "Unknown flag $1"
      exit 1
      ;;
    *)
      BUILD_TARGET="$1"
      shift
  esac
done


set -vex

BAZEL=${BAZEL:-bazel}  # Bazel version > 5.
PYTHON=${PYTHON:-python}  # Python, version should be > 3.7.

# Some environment variables used for XLA configure script, but set here anyway:
#if ((USE_GPU)) ; then
#  export TF_NEED_CUDA=1
#  export TF_CUDA_VERSION=12.3
#  export CUDA_VERSION=12.3
#  export TF_NEED_ROCM=0
#  export TF_CUDA_COMPUTE_CAPABILITIES="6.1,9.0"
#else
#  unset TF_NEED_CUDA
#fi
#export USE_DEFAULT_PYTHON_LIB_PATH=1
#export PYTHON_BIN_PATH=/usr/bin/python3
#export TF_CUDA_CLANG=0
#export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
#export CC_OPT_FLAGS=-Wno-sign-compare

# Check the OpenXLA version (commit hash) changed, and if so, download an
# updated `openxla_xla_bazelrc` file from github.
# TODO: include a sha256 verification of the file as well.
if ! egrep -q "^OPENXLA_XLA_COMMIT_HASH =" WORKSPACE ; then
  echo "Did not find OPENXLA_XLA_COMMIT_HASH in WORKSPACE file!?"
  exit 1
fi
OPENXLA_XLA_COMMIT_HASH="$(
  grep -E "^OPENXLA_XLA_COMMIT_HASH[[:space:]]*=[[:space:]]*" WORKSPACE |\
    sed -n 's/^[^"]*"\([^"]*\)".*/\1/p'
)"
printf "OPENXLA_XLA_COMMIT_HASH=%s\n" "${OPENXLA_XLA_COMMIT_HASH}"
OPENXLA_BAZELRC="openxla_xla_bazelrc"
if [[ ! -e "${OPENXLA_BAZELRC}" || ! -e "${OPENXLA_BAZELRC}.version" \
  || "$(< "${OPENXLA_BAZELRC}.version")" != "${OPENXLA_XLA_COMMIT_HASH}" ]] ; then
    echo "Fetching ${OPENXLA_BAZELRC} at version \"${OPENXLA_XLA_COMMIT_HASH}\""
    curl "https://raw.githubusercontent.com/openxla/xla/${OPENXLA_XLA_COMMIT_HASH}/.bazelrc" -o ${OPENXLA_BAZELRC}
    echo "${OPENXLA_XLA_COMMIT_HASH}" > "${OPENXLA_BAZELRC}.version"
else
    echo "File ${OPENXLA_BAZELRC} at version \"${OPENXLA_XLA_COMMIT_HASH}\" already exists, not fetching."
fi

STARTUP_FLAGS="${STARTUP_FLAGS} ${OUTPUT_DIR}"
STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=${OPENXLA_BAZELRC}"
STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=xla_configure.bazelrc"

# bazel build flags
BUILD_FLAGS="${BUILD_FLAGS:---keep_going --verbose_failures --sandbox_debug}"
BUILD_FLAGS="${BUILD_FLAGS} --config=linux"  # Linux only for now.
if ((DEBUG)) ; then
  BUILD_FLAGS="${BUILD_FLAGS} --config=dbg"
fi

# OpenXLA sets this to true for now to link with TF. But we need this enabled:
BUILD_FLAGS="${BUILD_FLAGS} --define tsl_protobuf_header_only=false"

# We need the dependencies to be linked statically -- they won't come from some external .so:
BUILD_FLAGS="${BUILD_FLAGS} --define framework_shared_object=false"


# XLA rules weren't meant to be exported, so we overrule their visibility
# constraints.
# BUILD_FLAGS="${BUILD_FLAGS} --check_visibility=false"

# Required from some `third_party/tsl` package:
BUILD_FLAGS="${BUILD_FLAGS} --experimental_repo_remote_exec"

# Attempts of enabling `cc_static_library`:
# See https://github.com/bazelbuild/bazel/issues/1920
# export USE_BAZEL_VERSION=last_green
# BUILD_FLAGS="${BUILD_FLAGS} --experimental_cc_static_library"
# Presumably, it will make to Bazel 7.4.0

# Required from more recent XLA bazel configuration.
# Whatever version is set here, XLA seems to require a matching "requirment_lock_X_YY.txt" file, where
# X=3, YY=11 match the python version.
export HERMETIC_PYTHON_VERSION=3.11

# Invoke bazel build
time "${BAZEL}" ${STARTUP_FLAGS} build ${BUILD_TARGET} ${BUILD_FLAGS} --build_tag_filters=-tfdistributed
