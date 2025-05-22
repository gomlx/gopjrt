# StableHLO Builder

> ** UNDER CONSTRUCTION **: Don't use this yet!

This package is a replacement for XlaBuilder, with the following advantages:

* XlaBuilder has become a second-class citizen so to say, within OpenXLA. And things are moving torwards
  the "MLIR builder" (MLIR is the generic ML Intermediary Language, of which StableHLO is an specialization/extension).
  So we will eventually need a newer "builder" for **Gopjrt**.
* Since PJRT takes StableHLO in plain text format, we can write this entirely in Go, not requiring any extra
  C/C++ library build. 
  * PJRT itself is a C library, but with a relatively small API surface, and for which
    there are prebuilt distributions available (for Jax). So we can get away without having to manage Bazel issues.
  * The goal is to eventually not require a C compiler to compile gopjrt, and instead
    use [ebitengine/purego](https://github.com/ebitengine/purego) do dynamically load PJRT.
  * There are PJRT for different platforms. If we don't need to compile XlaBuilder for them, it makes more plausible
    to support them.
 
The disadvantages:

* XlaBuilder provided "shape inference". So if I say `Add(a, b)` the XlaBuilder would tell how to broadcast
  a and b, and the resulting shape. When we build the StableHLO we have to re-implement this shape inference,
  not only for the `Gopjrt` users, but also because the *StableHLO* language requires the inputs and outputs shapes
  to be specified in every statement.
* This means more maintenance: any updates in the language specification or new ops need to have their shape inference
  updated accordingly.

## The `shapeinference` sub-package

The same code is also used by [**GoMLX**](github.com/gomlx/gomlx) `SimpleGo` engine 
(`github.com/gomlx/gomlx/backends/simplego`), but we didn't want to create a dependency in either direction:
users of **Gopjrt** may not be interested in **GoMLX**, and users of **GoMLX** that don't use the XLA backend
wouldn't want a dependency to **Gopjrt**. 

So the package `github.com/gomlx/gopjrt/stablehlo/shapeinference` is a copy of 
`github.com/gomlx/gomlx/backends/shapeinference`, with the later being the source of truth. We'll keep both in sync,
but if you need to change, please send a PR for that in [**GoMLX**](github.com/gomlx/gomlx).