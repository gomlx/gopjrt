# gopjrt

<img align="right" src="https://github.com/gomlx/gopjrt/assets/7460115/0f2869be-f64e-48b8-b2fa-1f6cbe703204" alt="Under Construction" width="480"/>

Go Wrappers for [OpenXLA PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt).

This is originally designed to power [GoMLX](github.com/gomlx/gomlx), but it may be used as a standalone, for lower level access to XLA, and other accelerator use cases.

## FAQ

* **Why is everything in one big package ?**
  Because of https://github.com/golang/go/issues/13467 : C api's cannot be exported across packages, even within the same repo. Even a function as simple as `func Add(a, b C.int) C.int` in one package cannot be called from another. So we need to wrap everything, and more than that, one cannot create separate sub-packages to handle separate concerns -- otherwise I would create one package for PJRT_Buffer, one for PJRT_Client, and so on. But alas ... they need to talk to each other, and more than that, pass C pointers around. One work around would be to pass `unsafe.Pointer` everywhere, and duplicate C helper functions ... but that would introduce yet other complications.
* **Why does PJRT spits out so much logging ? Can we disable it ?**
  This is a great question ... imagine if every library we use decided they too want too to clutter our stderr...
  Not sure why OpenXLA does that. They use [Abseil Logging](https://abseil.io/docs/python/guides/logging) which also has this other issue
  of not allowing two different linked programs/libraries to call its initialization (see [Issue #1656](https://github.com/abseil/abseil-cpp/issues/1656)).
  The only workaround I can think of is duplicating fd 2 and assign to Go's `os.Stderr`, and then close fd 2, so PJRT plugins won't have where to log --
  but I haven't tried, and this would impact other libraries that have legitimate use of `stderr`.

## Examples

**TODO**

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
