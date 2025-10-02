# GoPJRT ([Installing](#installing))

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gopjrt?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gopjrt)](https://github.com/Kwynto/gosession/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gopjrt)](https://goreportcard.com/report/github.com/gomlx/gopjrt)
[![TestStatus](https://github.com/gomlx/gopjrt/actions/workflows/go.yaml/badge.svg)](https://github.com/gomlx/gopjrt/actions/workflows/go.yaml)
![Coverage](https://img.shields.io/badge/Coverage-70.1%25-yellow)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)

## Why use GoPJRT ?

GoPJRT leverages [OpenXLA](https://openxla.org/) to compile, optimize and **accelerate numeric 
computations** (with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (NVidia, AMD ROCm*, Intel*, Apple Metal*) and TPU*. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

And because [Jax](https://docs.jax.dev/en/latest/), [TensorFlow](https://www.tensorflow.org/) and 
[optionally PyTorch](https://pytorch.org/xla/release/2.3/index.html) run on XLA, it is possible to run Jax functions in Go with GoPJRT, 
and probably TensorFlow and PyTorch as well.
See [example 2 in xlabuilder/README.md](https://github.com/gomlx/gopjrt/blob/main/xlabuilder/README.md#example-2).

(*) Not tested or partially supported by the hardware vendor.

GoPJRT aims to be minimalist and robust: it provides well-maintained, extensible Go wrappers for
[OpenXLA PJRT](https://openxla.org/#pjrt). 

GoPJRT is not very ergonomic (error handling everywhere), but it's expected to be a stable building block for
other projects to create a friendlier API on top. The same way [Jax](https://jax.readthedocs.io/en/latest/) is a Python friendlier API
on top of XLA/PJRT.

One such friendlier API co-developed with GoPJRT is [GoMLX, a Go machine learning framework](github.com/gomlx/gomlx).
But GoPJRT may be used as a standalone, for lower level access to XLA and other accelerator use cases—like running
Jax functions in Go, maybe an accelerated image processing or scientific simulation pipeline.

## What is what?

"**PJRT**" stands for "Pretty much Just another RunTime".

It is the heart of the OpenXLA project: it takes an IR (intermediate representation) of a "computation graph," JIT (Just-In-Time) compiles it
(once) and executes it fast (many times). 
See the [Google's "PJRT: Simplifying ML Hardware and Framework Integration"](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post.

A "computation graph" is the part of your program (usually vectorial math/machine learning related) that one
wants to "accelerate." 
It must be provided in an IR (intermediate representation) that is understood by the PJRT plugin. 
A few ways to create the computation graph IR:

1. [github.com/gomlx/stablehlo](https://github.com/gomlx/stablehlo?tab=readme-ov-file): [StableHLO](https://openxla.org/stablehlo)
is the current preferred IR language for XLA PJRT. This library (co-developed with **GoPJRT**) is a Go API for building
computation graphs in StableHLO, that can be directly fed to *GoPJRT*. See examples below.
2. [github.com/gomlx/gopjtr/xlabuilder](https://github.com/gomlx/gopjrt/tree/main/xlabuilder):
This is a wrapper Go library to an XLA C++ library that generates the previous IR (called MHLO).
It is still supported by XLA and by **GoPJRT**, but it is being deprecated.
3. Using Jax, Tensorflow, PyTorchXLA: Jax/Tensorflow/PyTorchXLA can output the StableHLO of JIT compiled functions, 
that can be fed directly to PJRT (as text). We don't detail this here, but the authors did this a lot during
development of **GoPJRT**, [github.com/gomlx/stablehlo](https://github.com/gomlx/stablehlo?tab=readme-ov-file) and 
[github.com/gomlx/gopjtr/xlabuilder](https://github.com/gomlx/gopjrt/tree/main/xlabuilder) for testing.

> [!NOTE]
> The IR (intermediary representation) that PJRT plugins accept are text, but not human-friendly to read/write.
> Small ones are debuggable, or can be used to probe which operations are being used behind the scenes,
> but definitely not friendly.

A "PJRT Plugin" is a dynamically linked library (`.so` file in Linux or `.dylib` in Darwin) that is able to JIT-compile
an IR of your computation graph and executes it for a particular hardware. So there are PJRT plugins 
for CPU (Linux/amd64 for now, and but likely it could be compiled for other CPUs -- SIMD/AVX is well-supported), 
for TPUs (Google's accelerator), 
GPUs (Nvidia is well-supported; there are AMD and Intel's PJRT plugins, but not tested)
and others are in development.

## Example

1. Minimalistic example, that assumes you have your StableHLO code in a variable (`[]byte`) called `stablehloCode`:

```go
var flagPluginName = flag.String("plugin", "cuda", "PRJT plugin name or full path")
...
plugin, err := pjrt.GetPlugin(*flagPluginName)
client, err := plugin.NewClient(nil)
executor, err := client.Compile().WithStableHLO(stablehloCode).Done()
for ii, value := range []float32{minX, minY, maxX, maxY} {
   inputs[ii], err = pjrt.ScalarToBuffer(m.client, value)
}
outputs, err := m.exec.Execute(inputs...).Done()
flat, err := pjrt.BufferToArray[float32](outputs[0])
outputs[0].Destroy() // Don't wait for the GC, destroy the buffer immediately.
...
```

2. See [mandelbrot.ipynb notebook](https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb) 
with an example building the computation for a Mandelbrot image using `stablehlo`, 
it includes a sample of the computation's StableHLO IR .

<a href="https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb">
<img src="https://github.com/gomlx/gopjrt/assets/7460115/d7100980-e731-438d-961e-711f04d4425e" style="width:400px; height:240px"/>
</a>

## How to use it ?

The main package is [`github.com/gomlx/gopjrt/pjrt`](https://pkg.go.dev/github.com/gomlx/gopjrt/pjrt), and we'll refer to it as simply `pjrt`.

The `pjrt` package includes the following main concepts:

* `Plugin`: represents a PJRT plugin. It is created by calling `pjrt.GetPlugin(name)` (where `name` is the name of the plugin).
  It is the main entry point to the PJRT plugin.
* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` a StableHLO program. It's the compiled/optimized/accelerated
  code ready to run.
* `Buffer`: Represents a buffer with the input/output data for the computations in the accelerators. There are 
  methods to transfer it to/from the host memory. They are the inputs and outputs of `LoadedExecutable.Execute`.

PJRT plugins by default are loaded after the program is started  (using `dlopen`). 
But there is also the option to pre-link the CPU PJRT plugin in your program. 
For that, just import (as `_`) one of the following packages:

- `github.com/gomlx/gopjrt/pjrt/cpu/static`: pre-link the CPU PJRT statically, so you don't need to distribute
  a CPU PJRT with your program. But it's slower to build, potentially taking a few extra (annoying) seconds
  (static libraries are much slower to link).
- `github.com/gomlx/gopjrt/pjrt/cpu/dynamic`: pre-link the CPU PJRT dynamically (as opposed to load it after the
  Go program starts). It is fast to build, but it still requires deploying the PJRT plugin along with your
  program. Not commonly used, but a possibility.

While it uses CGO to dynamically load the plugin and call its C API, `pjrt` doesn't require anything other than the plugin 
to be installed.

The project release includes pre-built CPU released for Linux/amd64 only now.
It's been compiled for Macs before -- I don't have easy access to a Apple Mac to maintain it.

It also includes a install program (see section **Installing** bellow) for Linux/CUDA PJRT and for Nvidia GPU support (
it uses the one from the Jax distributed binaries, extracted from Jax and Nvidia pip packages).


## Installing

GoPJRT requires a C library installed for XlaBuilder and one or more "PJRT plugin" modules (the thing that actually does the JIT compilation
of your computation graph). To facilitate, it provides an interactive and self-explanatory installer (it comes with lots of help messages):

```bash
go run github.com/gomlx/gopjrt/cmd/gopjt_installer
```

You can also provide directly the flags you want to avoid the interactive mode (so it can be used in Dockerfiles).

> [!NOTE]
> For now it only works for Linux/amd64 (or Windows+WSL) and NVidia CUDA. 
> I managed to write for Darwin(macOS) before, but not having easy access to a Mac to maintain it, eventually I dropped it.
> I would also love to support AMD ROCm, but again I don't have easy access to hardwre to test/maintain it.
> If you feel like contributing, or donating hardware/cloud credits, please contact me.
  
There are also some older bash install scripts under [`github.com/gomlx/gopjrt/cmd`](https://github.com/gomlx/gopjrt/tree/main/cmd),
but they are deprecated and eventually will be removed in a few versions. Let me know if you need them.

### Building C/C++ dependencies

If you want to build from scratch (both `xlabuilder` and `pjrt` dependencies), simply go to the `c/` subdirectory
and run `basel.sh`.
It uses [Bazel](https://bazel.build/) due to its dependencies to OpenXLA/XLA.
If not in one of the supported platforms, you will need to create a `xla_configure.OS_ARCH.bazelrc`
file.

## PJRT Plugins for other devices or platforms.

See [docs/devel.md](https://github.com/gomlx/gopjrt/blob/main/docs/devel.md#pjrt-plugins) on hints on how to compile a plugin 
from OpenXLA/XLA sources.

Also, see [this blog post](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html) with the link and references to the Intel and Apple hardware plugins. 

## FAQ

* **When is feature X from PJRT or XlaBuilder going to be supported ?**
  Yes, GoPJRT doesn't wrap everything—although it does cover the most common operations. 
  The simple ops and structs are auto-generated. But many require hand-writing.
  Please if it is useful to your project, create an issue, I'm happy to add it. I focused on the needs of GoMLX, 
  but the idea is that it can serve other purposes, and I'm happy to support it.
* **Why does PJRT spits out so much logging ? Can we disable it ?**
  This is a great question ... imagine if every library we use decided they also want to clutter our stderr?
  I have [an open question in Abseil about it](https://github.com/abseil/abseil-cpp/discussions/1700).
  It may be some issue with [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  A hacky workaround is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins
  won't have where to log. This hack is encoded in the function `pjrt.SuppressAbseilLoggingHack()`: just call it
  before calling `pjrt.GetPlugin`. But it may have unintended consequences, if some other library is depending
  on the fd 2 to work, or if a real exceptional situation needs to be reported and is not.

## 🤝 Collaborating or asking for help

Discussion in the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
(you can [join the slack server here](https://invite.slack.golangbridge.org/)).


## Environment Variables

That help control or debug how GoPJRT work:

* `PJRT_PLUGIN_LIBRARY_PATH`: Path to search for PJRT plugins. 
  GoPJRT also searches in `/usr/local/lib/gomlx/pjrt`, `${HOME}/.local/lib/gomlx/pjrt`, in
  the standard library paths for the system, and in the paths defined in `$LD_LIBRARY_PATH`.
* `XLA_FLAGS`: Used by the C++ PJRT plugins. Documentation is linked by the [Jax XLA_FLAGS page](https://docs.jax.dev/en/latest/xla_flags.html),
  but I found easier to just set this to "--help" and it prints out the flags.
* `XLA_DEBUG_OPTIONS`: If set, it is parsed as a `DebugOptions` proto that
  is passed during the JIT-compilation (`Client.Compile()`) of a computation graph.
  It is not documented how it works in PJRT (e.g. I observed a great slow down when this is set,
  even if set to the default values), but [the proto has some documentation](https://github.com/gomlx/gopjrt/blob/main/protos/xla.proto#L40).
* `GOPJRT_INSTALL_DIR` and `GOPJRT_NOSUDO`: used by the install scripts, see "Installing" section above.

## Links to documentation

* [Google Drive Directory with Design Docs](https://drive.google.com/drive/folders/18M944-QQPk1E34qRyIjkqDRDnpMa3miN): Some links are outdated or redirected, but very valuable information.
* [How to use the PJRT C API? #openxla/xla/issues/7038](https://github.com/openxla/xla/issues/7038): discussion of folks trying to use PJRT in their projects. Some examples leveraging some of the XLA C++ library.
* [How to use PJRT C API v.2 #openxla/xla/issues/7038](https://github.com/openxla/xla/issues/13733).
* [PJRT C API README.md](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md): a collection of links to other documents.
* [Public Design Document](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit).
* [Gemini](https://gemini.google.com) helped quite a bit parsing/understanding things -- despite the hallucinations -- other AIs may help as well.

## Running Tests

All tests support the following build tags to pre-link the CPU plugin (as opposed to `dlopen` the plugin) -- select at most one of them:  

* `--tags pjrt_cpu_static`: link (preload) the CPU PJRT plugin from the static library (`.a`) version. 
  Slowest to build (but executes the same speed).
* `--tags pjrt_cpu_dynamic`: link (preload) the CPU PJRT plugin from the dynamic library (`.so`) version. 
  Faster to build, but deployments require deploying the `libpjrt_c_api_cpu_dynamic.so` file along.

For Darwin (MacOS), for the time being it's hardcoded with static linking, so avoid using these tags. 

## Acknowledgements
This project utilizes the following components from the [OpenXLA project](https://openxla.org/):

* This project includes a (slightly modified) copy of the OpenXLA's [`pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) file. 
* OpenXLA PJRT CPU Plugin: This plugin enables execution of XLA computations on the CPU.
* OpenXLA PJRT CUDA Plugin: This plugin enables execution of XLA computations on NVIDIA GPUs.

* We gratefully acknowledge the OpenXLA team for their valuable work in developing and maintaining these plugins.

## Licensing:

GoPJRT is [licensed under the Apache 2.0 license](https://github.com/gomlx/gopjrt/blob/main/LICENSE).

The [OpenXLA project](https://openxla.org/), including `pjrt_c_api.h` file, the CPU and CUDA plugins, is [licensed under the Apache 2.0 license](https://github.com/openxla/xla/blob/main/LICENSE).

The CUDA plugin also utilizes the NVIDIA CUDA Toolkit, which is subject to NVIDIA's licensing terms and must be installed by the user.

For more information about OpenXLA, please visit their website at [openxla.org](https://openxla.org/), or the github page at [github.com/openxla/xla](https://github.com/openxla/xla)
