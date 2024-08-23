# Notes For Developers

## Package `pjrt`

The package tries to be independent of installation of any external compiled library -- including the other `xlabuilder` package.
It still depends on CGO and system libraries (`-ldl` for dynamic loading).

It includes a copy of the following files:

* `pjrt_c_api.h` from [github.com/openxla/xla/.../xla/pjrt/c/pjrt_c_api.h](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h), with the definitions of the PJRT plugin API.
  There is no easy way to integrate Go build system with Bazel (used by PJRT), so we just copied over the file (and mentioned it in the licensing).
* `compilation_options.proto`: ???

To generate the latest proto Go programs (see [tutorial](https://protobuf.dev/getting-started/gotutorial/)):
* Install the Protocol Buffers compiler: `sudo apt install protobuf-compiler`
* Install the most recent `protoc-gen-go`: `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`
* Set XLA_SRC to a clone of the `github.com/openxla/xla` repository.
* Go to the `protos` sub-package and do `go generate .` See `cmd/protoc_xla_prots/main.go` for details.

## Package `xlabuilder`

The package needs to link XLA's XlaBuilder library (and some associated tools). To achieve that, we create in the
subdirectory `c/` a C/C++ project that builds the `libgomlx_xlabuilder.so` file and associated header files. They
need to be installed so that `xlabuilder` package compile.

The plan is to have in the release a binary distribution of the required libraries. It's very inconvenient ... but
XLA is not easy to build (Bazel is complex and finicky) so trying to integrate everything would be tricky.

TODO: investigate distributing all headers and `libgomlx_xlabuilder.so` in the repository so it's fetched with Go,
and the user won't need to do anything.

## PJRT Plugins

* A prebuilt CUDA (GPU) plugin is  [distributed with Jax (pypi wheel)](https://pypi.org/project/jax-cuda12-pjrt/) (albeit with a [non-standard naming](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit#heading=h.l9ksu371j9wz))
* The CPU plugin can be built from the XLA sources: after running `configure.py`, build `bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so`.

## Updating `coverage.out` file

This is not done as a github actions because it would take too long to download the datasets, etc.
Instead, do it manually by running `cmd/run_coverage.sh` from the root of the repository.

