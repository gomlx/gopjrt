# gopjrt

Go Wrappers for [OpenXLA PjRT](https://github.com/openxla/xla/tree/main/xla/pjrt).

This is originally designed to power [GoMLX](github.com/gomlx/gomlx), but it may be used as a standalone, for lower level access to XLA, and other accelerator use cases.

## FAQ

* **Why is everything in one big package ?**
  Because of https://github.com/golang/go/issues/13467 : C api's cannot be exported accross packages, even within the same repo. Even a function as simple as `func Add(a, b C.int) C.int` in one package cannot be called from another. So we need to wrap everything, and more than that, one cannot create separate sub-packages to handle separate concerns -- otherwise I would create one package for PJRT_Buffer, one for PJRT_Client, and so on. But alas ... they need to talk to each other, and more than that, pass C pointers around. One work around would be to pass `unsafe.Pointer` everywhere, and duplicate C helper functions ... but that would introduce yet other complications.

## Examples

**TODO**

## Links to documentation

* [How to use the PJRT C API? #xla/issues/7038](https://github.com/openxla/xla/issues/7038): discussion of folks trying to use PjRT in their projects. The documentation is still lacking as of this writing.
* [PjRT C API README.md](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md): a collection of links to other documents.
* [Public Design Document](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit).
* [Gemini](https://gemini.google.com) helped quite a bit parsing/understanding things -- despite the hallucinations -- other AIs may help as well.

## Acknowledgements
This project utilizes the following components from the [OpenXLA project](https://openxla.org/):

* This project includes a copy of the OpenXLA's [`pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) file. 
* OpenXLA PjRT CPU Plugin: This plugin enables execution of XLA computations on the CPU.
* OpenXLA PjRT CUDA Plugin: This plugin enables execution of XLA computations on NVIDIA GPUs.
We gratefully acknowledge the OpenXLA team for their valuable work in developing and maintaining these plugins.

## Licensing:

**gopjrt** is [licensed under the Apache 2.0 license](https://github.com/gomlx/gopjrt/blob/main/LICENSE)

The [OpenXLA project](https://openxla.org/), including `pjrt_c_api.h` file, the CPU and CUDA plugins, is [licensed under the Apache 2.0 license](https://github.com/openxla/xla/blob/main/LICENSE).

The CUDA plugin also utilizes the NVIDIA CUDA Toolkit, which is subject to NVIDIA's licensing terms and must be installed by the user.

For more information about OpenXLA, please visit their website at [openxla.org](https://openxla.org/), or the github page at [github.com/openxla/xla](https://github.com/openxla/xla)
