#!/bin/bash

# This file will generate the Go generated code to handle Protocol Buffer messages used in XLA (including PJRT).
#
# It requires the github.com/openxla/xla repository to be cloned somewhere. Set XLA_SRC to that directory.

if [[ ! -d "${XLA_SRC}" || "${XLA_SRC}" = "" ]] ; then
  echo "Please set XLA_SRC to the directory containing the cloned github.com/openxla/xla repository somewhere." 1>&2
  exit 1
fi
set -e

# List of protos included.
protos=(
  "tsl/protobuf/dnn.proto"
  "xla/autotune_results.proto"
  "xla/autotuning.proto"
  "xla/pjrt/compile_options.proto"
  "xla/service/hlo.proto"
  "xla/stream_executor/device_description.proto"
  "xla/xla.proto"
  "xla/xla_data.proto"
)

# shellcheck disable=SC2168
go_opts=()
for p in "${protos[@]}" ; do
  go_opts+=("--go_opt=M${p}=github.com/gomlx/gopjrt/proto")
done

set -x
protoc --go_out=. -I="${XLA_SRC}" -I="${XLA_SRC}/third_party/tsl" \
  "--go_opt=module=github.com/gomlx/gopjrt/proto" \
  "${go_opts[@]}" \
  "${protos[@]}"

