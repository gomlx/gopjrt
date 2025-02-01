# gopjrt ([Installing](#installing))

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gopjrt?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gopjrt)](https://github.com/Kwynto/gosession/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gopjrt)](https://goreportcard.com/report/github.com/gomlx/gopjrt)
[![TestStatus](https://github.com/gomlx/gopjrt/actions/workflows/go.yaml/badge.svg)](https://github.com/gomlx/gopjrt/actions/workflows/go.yaml)
![Coverage](https://img.shields.io/badge/Coverage-69.8%25-yellow)

**`gopjrt`** leverages [OpenXLA](https://openxla.org/) to compile, optimize and accelerate numeric 
computations (with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (NVidia, Intel*, Apple Metal*) and TPU*. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

And because Jax, TensorFlow and [optionally PyTorch](https://pytorch.org/xla/release/2.3/index.html) run on XLA,
it is possible to run Jax functions in Go with `gopjrt`, and probably TensorFlow and PyTorch as well.
See example 2 below.

(*) Not tested or partially supported by the hardware vendor.

`gopjrt` aims to be minimalist and robust: it provides well maintained, extensible Go wrappers for
[OpenXLA PJRT](https://openxla.org/#pjrt) and [OpenXLA XlaBuilder](https://github.com/openxla/xla/blob/main/xla/client/xla_builder.h) libraries.

`gopjrt` is not very ergonomic (error handling everywhere), but it's expected to be a stable building block for
other projects to create a friendlier API on top. The same way [Jax](https://jax.readthedocs.io/en/latest/) is a friendlier API
on top of XLA/PJRT.

One such friendlier API is [GoMLX, a Go machine learning framework](github.com/gomlx/gomlx), but `gopjrt` may be used as a standalone, 
for lower level access to XLA and other accelerator use cases -- like running Jax functions in Go.

It provides 2 independent packages (often used together, but not necessarily):

## `github.com/gomlx/gopjrt/pjrt`: loading and using PJRT plugins

"**PjRt**" stands for "Pretty much Just another RunTime".

It is the heart of the OpenXLA project: it takes an IR (intermediate representation) of a "computation graph", JIT (Just-In-Time) compiles it
(once) and executes it (many times). 
See the [Google's "PJRT: Simplifying ML Hardware and Framework Integration"](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post.

A "computation graph" is the part of your program (usually vectorial math/machine learning related) that one
wants to "accelerate".

A "PJRT Plugin" is a dynamically linked library (`.so` file in Linux or `.dylib` in Darwin) that is able to JIT-compile
an IR of your computation graph (it is generated by `xlabuilder`, see the next section, so you don't need to worry about it),
and executes it for a particular hardware. So there are PJRT plugins for CPUs (Linux/Darwin amd64/arm64, and likely it could
be compiled for other CPUs -- SIMD/AVX is well-supported), for TPUs (Google's accelerator), GPUs (Nvidia is well-supported; there are AMD and Intel's
PJRT plugins, but not tested) and others may come.

The `github.com/gomlx/gopjrt/pjrt` package loads PJRT plugins dynamically 
(using [`dlopen`](https://man7.org/linux/man-pages/man3/dlopen.3.html)) in the form
of a dynamic linked library -- and provides an API to compile and execute "programs".

> [!NOTE]
> The IR (intermediary representation) that PJRT plugin accepts are "HLO" or "StableHLO".
> They not usually written directly by humans (except for debugging/developing the insides of PJRT),
> and instead are generated by the `xlabuilder` package described below, or in Python, by a Jax/PyTorch/Tensorflow 
> program.

The `pjrt` package includes the following main concepts:

* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` an HLO program. It's the compiled/optimized/accelerated
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
  program. Not many reasons to use this, but a possibility.

While it uses CGO to dynamically load the plugin and call its C API, `pjrt` doesn't require anything other than the plugin 
to be installed.

The project release includes pre-built CPU released for Linux/amd64, Darwin/arm64 (M1, M2, ... CPUs) and Darwin/amd64
(old i86_64 Macs). For Macs for now it only works by statically linking the CPU PJRT plugin (the default there).

It also includes a install script (see section **Installing** bellow) for Linux/CUDA PJRT, for Nvidia GPU support:
it uses the one from the Jax distributed binaries, extracted from Jax pip packages.

. But if one imports one of the packages 
`github.com/gomlx/gopjrt/pjrt/cpu/static` or  `github.com/gomlx/gopjrt/pjrt/cpu/dynamic`, the CPU plugin is
linked directly (statically or dynamically). This is slower to build, but in particular the static version
allows for a more convenient self-contained binary (except to the `libc` and `libstdc++` libraries, but those
are usually present everywhere).


## `github.com/gomlx/gopjrt/xlabuilder`

This provides a Go API for build accelerated computation using the [XLA Operations](https://openxla.org/xla/operation_semantics).
The output of building the computation using `xlabuilder` is an IR (intermediate representation, more specifically "HLO", 
but it doesn't matter here) that can be directly used with PJRT (and the `pjrt` package above).

Again it aims to be minimalist, robust and well maintained, albeit not very ergonomic necessarily.

Main concepts:

* `XlaBuilder`: builder object, used to keep track of the operations being added.
* `XlaComputation`: created with `XlaBuilder.Build(...)` and represents the finished program, ready to be used by 
  PJRT (or saved to disk). It is also used to represent sub-routines/functions -- see `XlaBuilder.CreateSubBuilder` and
  `Call` method.
* `Literal`: represents constants in the program. Some similarities with a `pjrt.Buffer`, but `Literal` is only used
  during the creation of the program. Usually, better to avoid large constants in a program, rather feed them
  as `pjrt.Buffer`, as inputs to the program during its execution.

See examples below.

The `xlabuilder` package includes a separate C project that generates a `libgomlx_xlabuilder.so` dynamic library 
(~13Mb for linux/x86-64) and associated `*.h` files, that need to be installed -- see **Installing** section below.
A `tar.gz` is included in the release for linux/amd64, darwin/arm64 and darwin/amd64 os/architectures. 
But one can also build it from scratch for different platforms -- it uses [Bazel](https://bazel.build/) due to its 
dependencies to OpenXLA/XLA.

Notice that there are alternatives to using `XlaBuilder`:

* JAX/TensorFlow can output the HLO of JIT compiled functions, that can be fed directly to PJRT (see example 2).
* Use [GoMLX](github.com/gomlx/gomlx).
* One can use `XlaBuilder` during development, and then save the output (see . And then
  during production only use the `pjrt` package to execute it.

## Examples

### [Example 1: Create function with XlaBuilder and execute it](https://github.com/gomlx/gopjrt/blob/main/gopjrt_test.go):

- This is a trivial example. XLA/PJRT really shines when doing large number crunching tasks.
- The package [`github.com/janpfeifer/must`](github.com/janpfeifer/must) simply converts errors to panics.

```go
  builder := xlabuilder.New("x*x+1")
  x := must.M1(xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32))) // Scalar float32.
  fX := must.M1(xlabuilder.Mul(x, x))
  one := must.M1(xlabuilder.ScalarOne(builder, dtypes.Float32))
  fX = must.M1(xlabuilder.Add(fX, one))

  // Get computation created.
  comp := must.M1(builder.Build(fX))
  //fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

  // PJRT plugin and create a client.
  plugin := must.M1(pjrt.GetPlugin(*flagPluginName))
  fmt.Printf("Loaded %s\n", plugin)
  client := must.M1(plugin.NewClient(nil))

  // Compile program.
  loadedExec := must.M1(client.Compile().WithComputation(comp).Done())
  fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)
	
  // Test values:
  inputs := []float32{0.1, 1, 3, 4, 5}
  fmt.Printf("f(x) = x^2 + 1:\n")
  for _, input := range inputs {
    inputBuffer := must.M1(pjrt.ScalarToBuffer(client, input))
    outputBuffers := must.M1(loadedExec.Execute(inputBuffer).Done())
    output := must.M1(pjrt.BufferToScalar[float32](outputBuffers[0]))
    fmt.Printf("\tf(x=%g) = %g\n", input, output)
  }

  // Destroy the client and leave -- alternatively it is automatically destroyed when garbage collected.
  must.M1(client.Destroy())
```
### Example 2: Execute Jax function in Go with `pjrt`:

First we create the HLO program in Jax/Python (see [Jax documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.xla_computation.html#jax.xla_computation))

_(You can do this with [Google's Colab](colab.resarch.google.com) without having to install anything)_

```python
import os
import jax

def f(x): 
  return x*x+1

comp = jax.xla_computation(f)(3.)
print(comp.as_hlo_text())
hlo_proto = comp.as_hlo_module()

with open('hlo.pb', 'wb') as file:
  file.write(hlo_proto.as_serialized_hlo_module_proto())
```

Then download the `hlo.pb` file and do:

- _(The package [`github.com/janpfeifer/must`](github.com/janpfeifer/must) simply converts errors to panics)_

```go
  hloBlob := must.M1(os.ReadFile("hlo.pb"))

  // PJRT plugin and create a client.
  plugin := must.M1(pjrt.GetPlugin(*flagPluginName))
  fmt.Printf("Loaded %s\n", plugin)
  client := must.M1(plugin.NewClient(nil))
  loadedExec := must.M1(client.Compile().WithHLO(hloBlob).Done())

  // Test values:
  inputs := []float32{0.1, 1, 3, 4, 5}
  fmt.Printf("f(x) = x^2 + 1:\n")
  for _, input := range inputs {
    inputBuffer := must.M1(pjrt.ScalarToBuffer(client, input))
    outputBuffers := must.M1(loadedExec.Execute(inputBuffer).Done())
    output := must.M1(pjrt.BufferToScalar[float32](outputBuffers[0]))
    fmt.Printf("\tf(x=%g) = %g\n", input, output)
  }
```

### Example 3: [Mandelbrot Set Notebook](https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb)

The notebook includes both the "regular" Go implementation and the corresponding implementation using `XlaBuilder` 
and execution with `PJRT` for comparison, with some benchmarks.

<a href="https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb">
<img src="https://github.com/gomlx/gopjrt/assets/7460115/d7100980-e731-438d-961e-711f04d4425e" style="width:400px; height:240px"/>
</a>

## Installing

**gopjrt** requires a C library installed and a "PJRT plugin" module (the thing that actually does the JIT compilation
of your computation graph). The following scripts install the required files under `/usr/local/lib` and `/usr/local/include`
by default (but they can be changed, see the scripts for all options).

### **TLDR;** Linux/amd64 (or Windows+WSL)

*For Linux or Windows+WSL*, run the following script ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_linux_amd64.sh)) to install under `/usr/local/{lib,include}`:

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash
```

To add CUDA (NVidia GPU) support, in addition run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_cuda.sh)):

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash
```

### **TLDR;** Darwin/arm64 (M1, M2, ... chips)

Currently, Darwin (MacOS) 🚧🛠 it only works with statically linked PJR CPU plugin 🛠🚧️ so that is the default (see issue in [XLA's issue #19152](https://github.com/openxla/xla/issues/19152) and on XLA's discord channels).
**Experimental**. 

It's slower to build (it adds a couple of seconds) but your program is statically linked and can more
easily be distributed (without needing to install the PJRT plugin).

To install run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_arm64.sh))):

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin_arm64.sh | bash
```

### **TLDR;** Darwin/amd64 (older i86_64 machines)

> [!WARNING]
> The Darwin pre-built binary versions is at least 2 versions behind the current one -- I don't have easily available
> mac box to update it. Some ops will not work. Since I can't test, there may be other things not working.
> 
> If you use Mac and are familiar with building Go and C/C++ programs, a contribution where would be very welcome! :smiley:
> Both for [GoMLX](https://github.com/gomlx/gomlx) and [Gopjrt](https://github.com/gomlx/gopjrt).

Currently, Darwin (MacOS) 🚧🛠 it only works with statically linked PJR CPU plugin 🛠🚧️ so that is the default (see issue in [XLA's issue #19152](https://github.com/openxla/xla/issues/19152) and on XLA's discord channels).
**Experimental**.

It's slower to build (it adds some seconds) but your program is statically linked and can more
easily be distributed (without needing to install the PJRT plugin).

To install run ([see source](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_arm64.sh))):

```bash
curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_darwin_amd64.sh | bash
```

### More details on installation scripts and PJRT plugins

The installation scripts 
[`cmd/install_linux_amd64.sh`](https://github.com/gomlx/gopjrt/blob/main/cmd/install_linux_amd64.sh),
[`cmd/install_cuda.sh`](https://github.com/gomlx/gopjrt/blob/main/cmd/install_cuda.sh),
[`cmd/install_darwin_arm64.sh`](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_arm64.sh) and
[`cmd/install_darwin_amd64.sh`](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_amd64.sh)
can be controlled to install in any arbitrary directory (by setting `GOPJRT_INSTALL_DIR`) and not to use `sudo` 
(by setting `GOPJRT_NOSUDO`).

You many need to set your `LD_LIBRARY_PATH` if the installation directory is not standard, and the `PJRT_PLUGIN_LIBRARY_PATH`
to tell gopjrt where to find the plugins -- if not using static linking.

There are two parts that needs installing: (1) XLA Builder library (it's a C++ wrapper) to write computation graphs; 
(2) PJRT plugins to execute graphs for the accelerator devices you want to support.

The releases come with prebuilt:

1. XLA Builder library for _linux/amd64_ (or Windows WSL), 
   _darwin/arm64_ and _darwin/amd64_. Both MacOS/Darwin releases are **EXPERIMENTAL** and have somewhat limited
   functionality (on the PJRT side), see https://developer.apple.com/metal/jax/.
2. The PJRT for CPU as a standalone plugin (can be preloaded/pre-linked or loaded on the fly with `dlopen`; and
   as a static library, that can be pre-linked: slower but more convenient for deployment. 

There is also the CUDA PJRT plugin downloader script, and the 
[`cmd/install_darwin_arm64.sh`](https://github.com/gomlx/gopjrt/blob/main/cmd/install_darwin_arm64.sh) script can
download the Apple/Metal PJRT Plugin if the environment variable `GOPJRT_METAL=1` 
(**EXPERIMENTAL** and not fully working, see notes in installation script).

The `gopjrt/pjrt` package if asked to load a PJRT plugin (see `pjrt.GetPlugin(name string)`) and it's not already 
loaded (or pre-linked), then it will search for PJRT plugins in `/usr/local/lib/gomlx/pjrt` and all standard library 
locations (configured in `/etc/ld.so.conf` in Linux).
Alternatively, one can set the directory(ies) to search for plugins setting the environment variable
`PJRT_PLUGIN_LIBRARY_PATH`.

There is always the option to pre-link (statically or dynamically) the CPU PJRT when building the Go program. See above.

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
  Yes, `gopjrt` doesn't wrap everything -- although it does cover the most common operations. 
  The simple ops and structs are auto-generated. But many require hand-writing.
  Please if it is useful to your project, create an issue, I'm happy to add it. I focused on the needs of GoMLX, 
  but the idea is that it can serve other purposes, and I'm happy to support it.
* **Why not split in smaller packages ?**
  Because of https://github.com/golang/go/issues/13467 : C API's cannot be exported across packages, even within the same repo.
  Even a function as simple as `func Add(a, b C.int) C.int` in one package cannot be called from another. 
  So we need to wrap everything, and more than that, one cannot create separate sub-packages to handle separate concerns.
  This is also the reason the library `chelper.go` is copied in both `pjrt` and `xlabuilder` packages.
* **Why does PJRT spits out so much logging ? Can we disable it ?**
  This is a great question ... imagine if every library we use decided they also want to clutter our stderr?
  I have [an open question in Abseil about it](https://github.com/abseil/abseil-cpp/discussions/1700).
  It may be some issue with [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  A hacky work around is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins
  won't have where to log. This hack is encoded in the function `pjrt.SuppressAbseilLoggingHack()`: just call it
  before calling `pjrt.GetPlugin`. But it may have unintended consequences, if some other library is depending
  on the fd 2 to work, or if a real exceptional situation needs to be reported and is not.

## Environment Variables

That help control or debug how **gopjrt** work:

* `PJRT_PLUGIN_LIBRARY_PATH`: Path to search for PJRT plugins. **gopjrt** also searches in `/usr/local/lib/gomlx/pjrt`,
  the standard library paths for the system and `$LD_LIBRARY_PATH`.
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

**gopjrt** is [licensed under the Apache 2.0 license](https://github.com/gomlx/gopjrt/blob/main/LICENSE).

The [OpenXLA project](https://openxla.org/), including `pjrt_c_api.h` file, the CPU and CUDA plugins, is [licensed under the Apache 2.0 license](https://github.com/openxla/xla/blob/main/LICENSE).

The CUDA plugin also utilizes the NVIDIA CUDA Toolkit, which is subject to NVIDIA's licensing terms and must be installed by the user.

For more information about OpenXLA, please visit their website at [openxla.org](https://openxla.org/), or the github page at [github.com/openxla/xla](https://github.com/openxla/xla)
