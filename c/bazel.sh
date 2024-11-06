#
# The following environment variables and flags can be defined:
#
# * TARGET_OS and TARGET_ARCH: they are set automatically to the running machine OS/ARCH, but can be set for
#   cross-compilation (experimental).
# * STARTUP_FLAGS and BUILD_FLAGS: passed as `bazel ${STARTUP_FLAGS} build <build_target> ${BUILD_FLAGS}.
# * --debug: Compile in debug mode, with symbols for gdb.
# * --output <dir>: Directory passed to `bazel --output_base`. Unfortunately, not sure why, bazel still outputs things
#   to $TEST_TMPDIR and /.cache.
# * <build_target>: Default is ":gomlx_xlabuilder_${TARGET_OS}_${TARGET_ARCH}".

# Versions 8 and above don't work. They seem to require blzmod (and the compatibility --enable_workspace build option
# doesn't seem to work the same):
# export USE_BAZEL_VERSION=last_green
export USE_BAZEL_VERSION=7.4.0  # First version allowing cc_static_library rule.

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


set -e

BAZEL=${BAZEL:-bazel}  # Bazel version > 7.4.
PYTHON=${PYTHON:-python}  # Python, version should be > 3.7.

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
BUILD_FLAGS="${BUILD_FLAGS:---keep_going --verbose_failures --sandbox_debug}"

# TARGET_OS and TARGET_ARCH defaults to current OS and architecture, but allows user to override the target platform
# (for cross-compilation).
if [[ -z "${TARGET_OS}" ]] ; then
  TARGET_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
fi
if [[ -z "${TARGET_ARCH}" ]] ; then
  TARGET_ARCH="$(uname -m)"
  if [[ "$TARGET_ARCH" == "x86_64" ]]; then
    TARGET_ARCH="amd64"
  elif [[ "$TARGET_ARCH" == "aarch64" ]]; then
    TARGET_ARCH="arm64"
  fi
fi
export TARGET_PLATFORM="${TARGET_OS}_${TARGET_ARCH}"
BUILD_TARGET="${BUILD_TARGET:-:gomlx_xlabuilder_${TARGET_PLATFORM}}"
BUILD_FLAGS="${BUILD_FLAGS} --action_env=TARGET_PLATFORM=${TARGET_PLATFORM} --define=TARGET_PLATFORM=${TARGET_PLATFORM}"
STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=xla_configure.${TARGET_PLATFORM}.bazelrc"

# Switch statement for TARGET_PLATFORM.
case "${TARGET_PLATFORM}" in
  "linux_amd64")
    echo "Building for Linux amd64"
    BUILD_FLAGS="${BUILD_FLAGS} --config=linux"
    ;;

  "darwin_amd64")
    echo "Building for macOS amd64"
    STARTUP_FLAGS="${STARTUP_FLAGS} --bazelrc=custom_darwin_amd64.bazelrc"
    BUILD_FLAGS="${BUILD_FLAGS} --config=macos_amd64"
    # Apple/Metal PJRT only works with StableHLO, so we link it along.
    BUILD_FLAGS="${BUILD_FLAGS} --define use_stablehlo=false"
    ;;

  "darwin_arm64")
    echo "Building for macOS arm64"
    BUILD_FLAGS="${BUILD_FLAGS} --config=macos_arm64"
    # Apple/Metal PJRT only works with StableHLO, so we link it along.
    BUILD_FLAGS="${BUILD_FLAGS} --define use_stablehlo=true"
    ;;

  *)
    echo "Unsupported TARGET_PLATFORM: ${TARGET_PLATFORM}"
    exit 1
    ;;
esac

echo "TARGET_PLATFORM auto-detected: ${TARGET_PLATFORM}"
echo "Building for ${TARGET_PLATFORM}"

# Debug flags.
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
# Presumably, it will make to Bazel 7.4.0
BUILD_FLAGS="${BUILD_FLAGS} --experimental_cc_static_library"

# Absl and TSL redefine each others macros, and the output is full of garbage.
BUILD_FLAGS="${BUILD_FLAGS} --cxxopt=-Wno-macro-redefined"

# Required from more recent XLA bazel configuration.
# Whatever version is set here, XLA seems to require a matching "requirment_lock_X_YY.txt" file, where
# X=3, YY=11 match the python version.
export HERMETIC_PYTHON_VERSION=3.11

# Invoke bazel build
set -vx
time "${BAZEL}" ${STARTUP_FLAGS} build ${BUILD_TARGET} ${BUILD_FLAGS} --build_tag_filters=-tfdistributed
