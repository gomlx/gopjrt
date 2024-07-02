# gopjrt

<img align="right" src="https://github.com/gomlx/gopjrt/assets/7460115/0f2869be-f64e-48b8-b2fa-1f6cbe703204" alt="Under Construction" width="480"/>

**`gopjrt`** leverages [OpenXLA](https://openxla.org/) to make it easy to compile, optimize and accelerate numeric 
computations (with large data) from Go using various [backends supported by OpenXLA](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html): CPU, GPUs (NVidia, Intel*, Apple Metal*) and TPU*. 
It can be used to power Machine Learning frameworks (e.g. [GoMLX](github.com/gomlx/gomlx)), image processing, scientific 
computation, game AIs, etc. 

(*) Not tested yet, pls let me know if it works for you, or if you can lend access to these hardware (a virtual machine)
so that I can use (a virtual machine) for a while, I would love to try to verify and make sure it works there.

`gopjrt` aims to be minimalist and robust: provide a well maintained, extensible Go wrappers for
[OpenXLA PJRT](https://openxla.org/#pjrt) and [OpenXLA XlaBuilder](https://github.com/openxla/xla/blob/main/xla/client/xla_builder.h) libraries.
It is not very ergonomic (error handling everywhere), and the expectation is that others will create a 
friendlier API on top of `gopjrt` -- the same way [Jax](https://jax.readthedocs.io/en/latest/) is a friendlier API
on top of XLA/PJRT.

One such friendlier API is [GoMLX, a Go machine learning framework](github.com/gomlx/gomlx), but `gopjrt` may be used as a standalone, for lower level access to XLA, 
and other accelerator use cases.

It provides 2 independent packages (often used together, but not necessarily):

## `github.com/gomlx/gopjrt/pjrt`

This package loads [_PJRT plugins_](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html) -- implementations of PJRT for specific hardware (CPU, GPUs, TPUs, etc.) in the form
of a dynamic linked library -- and provides an API to compile and execute "programs".

"Programs" for PJRT are specified as "StableHLO serialized proto-buffers" (`HloModuleProto` more specifically). 
This is an intermediary (not usually written directly by humans) representation (IR) that can be output by, 
for instance, a Jax/PyTorch/Tensorflow program, or using the `xlabuilder` package, the library that follows.

It includes the following main concepts:

* `Client`: first thing created after loading a plugin. It seems one can create a singleton `Client` per plugin,
  it's not very clear to me why one would create more than one `Client`.
* `LoadedExecutable`: Created when one calls `Client.Compile` an HLO program. It's the compiled/optimized/accelerated
  code ready to run.
* `Buffer`: Represents a buffer with the input/output data for the computations in the accelerators. There are 
  methods to transfer it to/from the host memory. They are the inputs and outputs of `LoadedExecutable.Execute`.

**[Simple example](https://github.com/gomlx/gopjrt/blob/main/gopjrt_test.go):**

* **Note**: this is a trivial example. XLA/PJRT really shines when doing large number crunching tasks.

```go
	builder := xlabuilder.New("x*x+1")
	x, err := xlabuilder.Parameter(builder, "x", 0, xlabuilder.MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err, "Failed to create Parameter")
	fX, err := xlabuilder.Mul(x, x)
	require.NoError(t, err, "Failed operation Mul")
	one, err := xlabuilder.Constant(builder, xlabuilder.NewScalarLiteral(float32(1)))
	require.NoError(t, err, "Failed to create constant of 1")
	fX, err = xlabuilder.Add(fX, one)
	require.NoError(t, err, "Failed operation Add")

	// Get computation created.
	comp, err := builder.Build(fX)
	require.NoError(t, err, "Failed to build XlaComputation from ops.")
	//fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

	// PJRT plugin and create a client.
	plugin, err := pjrt.GetPlugin(*flagPluginName)
	require.NoError(t, err, "Failed to get plugin %q", *flagPluginName)
	fmt.Printf("Loaded %s\n", plugin)
	client, err := plugin.NewClient(nil)
	require.NoErrorf(t, err, "Failed to create a client on %s", plugin)
	fmt.Printf("	client: %s\n", client)

	// Compile program.
	loadedExec, err := client.Compile().WithComputation(comp).Done()
	require.NoErrorf(t, err, "Failed to compile our x^2 HLO program")
	fmt.Printf("Compiled program: name=%s, #outputs=%d\n", loadedExec.Name, loadedExec.NumOutputs)

	// Test values:
	inputs := []float32{0.1, 1, 3, 4, 5}
	wants := []float32{1.01, 2, 10, 17, 26}
	fmt.Printf("f(x) = x^2 + 1:\n")
	for ii, input := range inputs {
		// Transfer input to a on-device buffer.
		inputBuffer, err := pjrt.ScalarToBuffer(client, input)
		require.NoErrorf(t, err, "Failed to create on-device buffer for input %d", input)

		// Execute: it returns the output on-device buffer(s).
		outputBuffers, err := loadedExec.Execute(inputBuffer).Done()
		require.NoErrorf(t, err, "Failed to execute on input %d", input)

		// Transfer output on-device buffer to a "host" value (in Go).
		output, err := pjrt.BufferToScalar[float32](outputBuffers[0])
		require.NoErrorf(t, err, "Failed to transfer results of execution on input %d", input)

		// Print an check value is what we wanted.
		fmt.Printf("\tf(x=%g) = %g\n", input, output)
		require.InDelta(t, output, wants[ii], 0.001)
	}

	// Destroy the client and leave.
	err = client.Destroy()
	require.NoErrorf(t, err, "Failed to destroy client on %s", plugin)
```

While it uses CGO to dynamically load the plugin and call its C API, it doesn't require anything other than the plugin 
to be installed.

The project releases includes 2 plugins, one for CPU (linux-x86) compiled from XLA source code, and one for GPUs
provided in the Jax distributed binaries. But there are instructions to build your own CPU plugin (e.g.: for a different
architecture), or GPU (XLA seems to have code to support ROCm, but I'm not sure of the status). 
And it should work with binary plugins provided by other companies (Google Cloud will have a TPU PJRT 
plugin in their cloud TPU boxes; I hear Intel also have binary plugins for their hardware).

See plugins references in [PJRT blog post](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html).

## `github.com/gomlx/gopjrt/xlabuilder`

This provides a Go API for build accelerated computation using the [XLA Operations](https://openxla.org/xla/operation_semantics).
The output of building the computation using `xlabuilder` is an [_StableHLO(-ish)_](https://openxla.org/stablehlo)
program that can be directly used with PJRT (and the `pjrt` package above).

It aims to be minimalist and robust: provide a well maintained, extensible Go wrappers. 
But it is not very ergonomic (error handling everywhere), and the expectation is that others will create a
friendlier API on top of `gopjrt` -- the same way [Jax](https://jax.readthedocs.io/en/latest/) is a friendlier API
on top of XLA/PJRT. See [GoMLX](github.com/gomlx/gomlx) for such a friendlier interface.

Main concepts:

* `XlaBuilder`: builder object, used to keep track of the operations being added.
* `XlaComputation`: created with `XlaBuilder.Build(...)` and represents the finished program, ready to be used by 
  PJRT (or saved to disk). It is also used to represent sub-routines/functions -- see `XlaBuilder.CreateSubBuilder` and
  `Call` method.
* `Literal`: represents constants in the program. Some similarities with a `pjrt.Buffer`, but `Literal` is only used
  during the creation of the program. Usually, better to avoid large constants in a program, rather feed them
  as `pjrt.Buffer`, as inputs to the program during its execution.

**See example above, or check the tests, they provide good examples.**

The `xlabuilder` package includes a separate C project that generates a `libgomlx_xlabuilder.so` dynamic library 
(~13Mb for linux/x86-64) and associated `*.h` files, that need to be included/available. The binary of the library
is included, but one can also build it from scratch for different platforms -- it uses [Bazel](https://bazel.build/)
due to its dependencies to OpenXLA/XLA.

Notice that there are alternatives to using `XlaBuilder`:

* JAX/TensorFlow can output the HLO of JIT compiled functions, that can be fed directly to PJRT (TODO: write a small example/tutorial).
* Use [GoMLX](github.com/gomlx/gomlx).
* One can use `XlaBuilder` during development, and then save the output (see `XlaComputation.SerializedHLO`). And then
  during production only use the `pjrt` package to execute it.

## FAQ

* **Why is [GoMLX](github.com/gomlx/gomlx) is not using `gopjrt` ?**
  Not yet, soon.
* **When is feature X from PJRT or XlaBuilder going to be supported ?**
  Yes, `gopjrt` doesn't wrap everything -- the simple ops and structs are auto-generated. But many require hand-writing.
  Please if it is useful to your project, create an issue, I'm happy to add it. I focused on the needs of GoMLX, 
  but the idea is that it can server other purposes, and I'm happy to support it.
* **Why not split in smaller packages ?**
  Because of https://github.com/golang/go/issues/13467 : C API's cannot be exported across packages, even within the same repo.
  Even a function as simple as `func Add(a, b C.int) C.int` in one package cannot be called from another. 
  So we need to wrap everything, and more than that, one cannot create separate sub-packages to handle separate concerns.
  THis is also the reason the library `chelper.go` is copied in both `pjrt` and `xlabuilder` packages.
* **Why does PJRT spits out so much logging ? Can we disable it ?**
  This is a great question ... imagine if every library we use decided they also want to clutter our stderr...
  Not sure why OpenXLA does that. They use [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  The only workaround I can think of is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins won't have where to log --
  but I haven't tried, and this would impact other libraries that have legitimate use of `stderr`.

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
