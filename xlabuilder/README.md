## `github.com/gomlx/gopjrt/xlabuilder`: creating programs to be executed with PJRT

XlaBuilder is a Go wrapper around the corresponding XLA C++ library, which generates HLO (High Level Optimized)
IR for PJRT.

> [!Note]
> XlaBuilder is in maintenance mode by XLA in favor of [StableHLO](https://openxla.org/stablehlo), the current recommended IR for PJRT.
> Likewise, we are deprecating this library in favor of `github.com/gomlx/stablehlo`(https://github.com/gomlx/stablehlo),
> a Go API that generates StableHLO directly.
> The CPU PJRT still accepts both languages, but some PJRT (like Apple Metal) accepts only StableHLO.
>
> GoMLX is also moving to use `stablehlo`

This provides a Go API for build accelerated computation using the [XLA Operations](https://openxla.org/xla/operation_semantics).
The output of building the computation using `xlabuilder` is an IR (intermediate representation, more specifically "MHLO",
but it doesn't matter here) that can be directly used with PJRT (and the `pjrt` package above).

It aims to be minimalist, robust and well maintained, albeit not very ergonomic necessarily.

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

## Environment variables

* `GOPJRT_NO_STABLE_HLO`: If StableHLO is linked into `xlabuilder`, it will by default try to send StableHLO
  to the PJRT. Except if this is set, in which case it will continue using `HLOModule proto` to send to PJRT.
* `GOPJRT_TEXT_STABLE_HLO`: If not empty, and if StableHLO is linked into `xlabuilder`, it forces `xlabuilder`
  to use a textual StableHLO representation of the programs with PJRT.
  Use it in combination with `-vmodule=compile=2` flag to enable logging of each program compiled.

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
