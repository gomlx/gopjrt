# gopjrt

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/gopjrt?tab=doc)
[![GitHub](https://img.shields.io/github/license/gomlx/gopjrt)](https://github.com/Kwynto/gosession/blob/master/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gomlx/gopjrt)](https://goreportcard.com/report/github.com/gomlx/gopjrt)
[![TestStatus](https://github.com/gomlx/gopjrt/actions/workflows/go.yaml/badge.svg)](https://github.com/gomlx/gojrt/actions/workflows/go.yaml)
![Coverage](https://img.shields.io/badge/Coverage-00.0%25-yellow)

**`gopjrt`** leverages [OpenXLA](https://openxla.org/) to compile, optimize and accelerate numeric 
computations (with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (NVidia, Intel*, Apple Metal*) and TPU*. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

And because Jax, TensorFlow and [optionally PyTorch](https://pytorch.org/xla/release/2.3/index.html) run on XLA,
it is possible to run Jax functions in Go with `gopjrt`, and probably TensorFlow and PyTorch as well.
See example 2 below.

(*) Not tested yet, pls let me know if it works for you, or if you can lend access to these hardware (a virtual machine)
so that I can use (a virtual machine) for a while, I would love to try to verify and make sure it works there.

`gopjrt` aims to be minimalist and robust: it provides well maintained, extensible Go wrappers for
[OpenXLA PJRT](https://openxla.org/#pjrt) and [OpenXLA XlaBuilder](https://github.com/openxla/xla/blob/main/xla/client/xla_builder.h) libraries.

It is not very ergonomic (error handling everywhere), and the expectation is that others will create a 
friendlier API on top of `gopjrt` -- the same way [Jax](https://jax.readthedocs.io/en/latest/) is a friendlier API
on top of XLA/PJRT.

One such friendlier API is [GoMLX, a Go machine learning framework](github.com/gomlx/gomlx), but `gopjrt` may be used as a standalone, 
for lower level access to XLA and other accelerator use cases -- like running Jax functions in Go.

It provides 2 independent packages (often used together, but not necessarily):

## `github.com/gomlx/gopjrt/pjrt`

This package loads [_PJRT plugins_](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html) -- implementations of PJRT for specific hardware (CPU, GPUs, TPUs, etc.) in the form
of a dynamic linked library -- and provides an API to compile and execute "programs".

"Programs" for PJRT are specified as "StableHLO serialized proto-buffers" (`HloModuleProto` more specifically). 
This is an intermediary representation (IR) not usually written directly by humans that can be output by, 
for instance, a Jax/PyTorch/Tensorflow program, or using the `xlabuilder` package described below.

It includes the following main concepts:

* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` an HLO program. It's the compiled/optimized/accelerated
  code ready to run.
* `Buffer`: Represents a buffer with the input/output data for the computations in the accelerators. There are 
  methods to transfer it to/from the host memory. They are the inputs and outputs of `LoadedExecutable.Execute`.

While it uses CGO to dynamically load the plugin and call its C API, `pjrt` doesn't require anything other than the plugin 
to be installed.

The project release includes 2 plugins, one for CPU (linux-x86) compiled from XLA source code, and one for GPUs
provided in the Jax distributed binaries -- both for linux/x86-64 architecture (help with Mac wanted!).
But there are instructions to build your own CPU plugin (e.g.: for a different
architecture), or GPU (XLA seems to have code to support ROCm, but I'm not sure of the status). 
And it should work with binary plugins provided by others -- see plugins references in [PJRT blog post](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html).


## `github.com/gomlx/gopjrt/xlabuilder`

This provides a Go API for build accelerated computation using the [XLA Operations](https://openxla.org/xla/operation_semantics).
The output of building the computation using `xlabuilder` is an [_StableHLO(-ish)_](https://openxla.org/stablehlo)
program that can be directly used with PJRT (and the `pjrt` package above).

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
(~13Mb for linux/x86-64) and associated `*.h` files, that need to be installed. A `tar.gz` is included in the release
for linux/x86-64 architecture (help for Macs wanted!). 
But one can also build it from scratch for different platforms -- it uses [Bazel](https://bazel.build/) due to its dependencies to OpenXLA/XLA.

Notice that there are alternatives to using `XlaBuilder`:

* JAX/TensorFlow can output the HLO of JIT compiled functions, that can be fed directly to PJRT (see example 2).
* Use [GoMLX](github.com/gomlx/gomlx).
* One can use `XlaBuilder` during development, and then save the output (see `XlaComputation.SerializedHLO`). And then
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

  // Destroy the client and leave.
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

There are two parts that needs installing:

### Installing PJRT plugins

The recommended location for plugins is `/usr/local/lib/gomlx/pjrt`, but the `pjrt` package
will automatically search in all standard library locations (configured in `/etc/ld.so.conf`).
Alternatively, one can set the directory(ies) to search for plugins setting the environment variable
`PJRT_PLUGIN_LIBRARY_PATH`.

#### CPU for Linux

The release comes with a CPU plugin pre-compiled for the _linux/x86-64_ platform. The file is called
`pjrt_c_api_cpu_plugin.so.gz`. Please, uncompress the file and move it to your plugin directory -- e.g.:
`/usr/local/lib/gomlx/pjrt`.

#### NVidia's CUDA for Linux

NVidia licenses are complicated (I don't understand), so ... I hesitate to provide a prebuilt plugin and dependencies.
But there is a simple way to achieve it, by linking the files from a Jax installation.

Create and activate a [virtual environment (venv) for Python](https://docs.python.org/3/library/venv.html).
Probably a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
would also work.

Then install Jax for Cuda and its dependencies:

```shell
pip install -U "jax[cuda12]"
```

Now we want to link 2 things: (1) the cuda PJRT plugin; (2) the various NVidia drivers. 
Assuming the virtual environment (Python's `venv`) is set, the `$VIRTUAL_ENV` should be pointing
to its installation. Check that is the case with `$VIRTUAL_ENV`.

And then do (change the target directories to your preference):

```shell
sudo mkdir -p /usr/local/lib/gomlx/pjrt
sudo ln -sf ${VIRTUAL_ENV}/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
  /usr/local/lib/gomlx/pjrt/pjrt_c_api_cuda_plugin.so
sudo ln -sf ${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia \
  /usr/local/lib/gomlx/nvidia

```

#### Metal Mac

Should be doable, in a similar way as but I don't own a Mac. Contributions would be most welcome.

#### Plugins for other devices or platforms.

See [docs/devel.md](https://github.com/gomlx/gopjrt/blob/main/docs/devel.md#pjrt-plugins) on hints on how to compile a plugin 
from OpenXLA/XLA sources.

Also, see [this blog post](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html) with the link and references to the Intel and Apple hardware plugins. 


### Installing XlaBuilder C/C++ library (for Linux only for now)

This is only required is the XlaBuilder library (`xlabuilder` package) is used.

The release comes with a CPU plugin pre-compiled for the _linux/x86-64_ platform. The file is called
`gomlx_xlabuilder-linux-amd64.tar.gz` and it includes two subdirectories `lib/` and `include/` with the files
required to compile Go's `xlabuilder` package.

The suggest location is to "untar" (decompress and unpackage) this file to `/usr/local`.
Change the path to the file on the command below:

```shell
cd /usr/local
sudo tar xzvf gomlx_xlabuilder-linux-amd64.tar.gz
```

Finally, you want to make sure that your environment variable `LD_LIBRARY_PATH` includes `/usr/local/lib`.
Or that your system library paths in `/etc/ld.so.conf` include `/usr/local/lib`. 

## FAQ

* **Why is [GoMLX](github.com/gomlx/gomlx) is not using `gopjrt` ?**
  Not yet, soon.
* **When is feature X from PJRT or XlaBuilder going to be supported ?**
  Yes, `gopjrt` doesn't wrap everything -- although it does cover the most common operations. 
  The simple ops and structs are auto-generated. But many require hand-writing.
  Please if it is useful to your project, create an issue, I'm happy to add it. I focused on the needs of GoMLX, 
  but the idea is that it can serve other purposes, and I'm happy to support it.
* **Why not split in smaller packages ?**
  Because of https://github.com/golang/go/issues/13467 : C API's cannot be exported across packages, even within the same repo.
  Even a function as simple as `func Add(a, b C.int) C.int` in one package cannot be called from another. 
  So we need to wrap everything, and more than that, one cannot create separate sub-packages to handle separate concerns.
  THis is also the reason the library `chelper.go` is copied in both `pjrt` and `xlabuilder` packages.
* **Why does PJRT spits out so much logging ? Can we disable it ?**
  This is a great question ... imagine if every library we use decided they also want to clutter our stderr?
  I have [an open question in Abseil about it](https://github.com/abseil/abseil-cpp/discussions/1700).
  It may be some issue with [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  A hacky work around is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins
  won't have where to log. This hack is encoded in the function `pjrt.SuppressAbseilLoggingHack()`: just call it
  before calling `pjrt.GetPlugin`. But it may have unintended consequences, if some other library is depending
  on the fd 2 to work, or if a real exceptional situation needs to be reported and is not.

## Links to documentation

* [Google Drive Directory with Design Docs](https://drive.google.com/drive/folders/18M944-QQPk1E34qRyIjkqDRDnpMa3miN): Some links are outdated or redirected, but very valuable information.
* [How to use the PJRT C API? #openxla/xla/issues/7038](https://github.com/openxla/xla/issues/7038): discussion of folks trying to use PJRT in their projects. Some examples leveraging some of the XLA C++ library.
* [How to use PJRT C API v.2 #openxla/xla/issues/7038](https://github.com/openxla/xla/issues/13733).
* [PJRT C API README.md](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md): a collection of links to other documents.
* [Public Design Document](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit).
* [Gemini](https://gemini.google.com) helped quite a bit parsing/understanding things -- despite the hallucinations -- other AIs may help as well.

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
