# v0.4.9 - 2024-11-25

* Optional preloading CPU PJRT plugin:
  * `github.com/gomlx/gopjrt/pjrt/cpu/static` that statically links the PJRT CPU plugin: easy to deploy binary. 
    It includes the corresponding C BUILD rule to build the static library (`libpjrt_c_api_cpu_static.a`)
  * `github.com/gomlx/gopjrt/pjrt/cpu/dynamic` that dynamically links (and preloads) the PJRT CPU plugin.
* `pjrt_c_api_cpu.so` now compiled directly from `gopjrt`, and doesn't require cloning `xla` separately. It will
  be distributed in the same `tar.gz` file.
* Added MacOS support by statically linking the CPU PJRT plugin.

# v0.4.8 - 2024-11-19

* Replaced C++ `xla::StatusOr` by `absl::StatusOr` (the former was already an alias to the later).

# v0.4.7 - 2024-11-17

* Sync'ed with updated proto definitions from OpenXLA/XLA project.
* TestEndToEnd: added `klog` flags; list devices before trying to compile. 
* Renamed deprecated xla::Status to absl::Status.
* Update to XLA and PJRT v0.57
  * Updated XLA dependency.
  * Updated PJRT CPU plugin.
  * Updated `pjrt_c_api.h`: copying over from XLA source is now part of the generate program.
  * Note: PJRT v0.56 was broken for a few days, and the version was skipped.
    (breakage here https://github.com/openxla/xla/commit/590b36f89d8cb038e9e3929aeaea6e60451ef3fc#r149134910)
* **Mac version broken** :( : Following up on https://github.com/openxla/xla/issues/19152. Since it's
  outside our control, not blocking the release here.

# v0.4.6

* Fix to installation script: missing `sudo` to remove old library, not observing the GOPJRT_NOSUDO request.
* Fixed github test action `go.yaml`.
* Explicitly set the random algorithm to Philox when using RngBitGenerator. Also improved documentation and added
  check on the validity of the random state shape.
* Added `dtype.DType.IsUnsigned()`

# v0.4.5 

* Fixes to experimental/GPU MacOS (darwin) on arm64.
* XlaBuilder works on Darwin/X86_64 (darwin_amd64) but OpenXLA/XLA PJRT CPU does not work (yet?).
* Normalized names of prebuilt-binaries.
* Test `TestEndToEnd` only test first device by default, because CPU PJRT seems to falsely advertise more than one addressable device.
  * Added `--alldevices` to loop over all devices during the test.

# v0.4.4 - 2024-10-24

* Package `pjrt`: 
  * Fixed some API documentation issues with Buffer transfers from host.
* Package `xlabuilder`:
  * Fixed `NewArrayLiteral[T dtypes.Supported](flat []T, dimensions ...int)` to create a scalar if no dimensions are passed.

# v0.4.3 - 2024-10-23

* GoMLX XlaBuilder C library is now linked as a static library (`.a` instead of `.so`).
  * Using new Bazel 7.4.0, with support for `cc_static_library`.
* **EXPERIMENTAL** support for Apple/Metal (`darwin-arm64`) support:
  * Added C-wrapper compilation for darwin-arm64.
  * Added converter from HLO to StableHLO -- it greatly increases the size of libgomlx_builder.a, since it has to
    include the whole LLVM :(
    * Enables Apple Metal PJRT -- it only supports StableHLO/MLIR programs (and not the simpler HLO).
    * Only enabled for Darwin
* Updated XLA dependency; Updated PJRT for linux/amd64 CPU.
* Added `Literal.Data()`

# v0.4.2 - 2024-10-03

* Added `IsFinite` and `PopulationCount` operations.

# v0.4.1 - 2024-09-28

* Added memory layout information in buffer-to-host transfers: required for TPU.
* Included C error message when reporting PJRT plugin failures.
* Added GOPJRT_NOSUDO and GOPJRT_INSTALL_DIR to control `cmd/install.sh` and `cmd/install_cuda.sh`.
* Improved installation instructions to install directly from Github using `curl`, without the need to clone the repository.
* Updated `XlaBuilder` C-wrapper to refactorings withing github.com/openxla/xla.

# v0.4.0 - 2024-09-23

* Binary distributed compiled in Ubuntu 24.04 (glibc 2.38), updated dependencies on the C library. This may cause issues in older distributions.
* Added Erf operation.
* Added dtypes.MapOfNames that includes its aliases.
* Updated binary PJRT CPU plugin build, 50% faster in some cases (!)

# v0.3.2

* Added ReduceAnd and ReduceOr logical operations.

# v0.3.1

* Fixed +/-Inf for bfloat16.
* Removed dependencies on "github.com/gomlx/exceptions".

# v0.3.0 Some of the API now returns errors instead of panic

* Moved each compiled XLA proto to their own package under `gopjrt/protos/`: this facilitates conversion to Google3 BUILD scheme.
* Converted several panics to error returning from pjrt and xlabuilder. This means the API changed a bit.
* Added script `cmd/run_coverage.sh`.

# v0.2.4

* Added bfloat16 support.

# v0.2.3

* Fixed check for Nvidia GPU cards so it works within docker images.

# v0.2.2

* Added `install.sh` and `install_cuda.sh`
* `pjrt.AvailablePlugins` now checks that the plugin can be initialized: so if a "cuda" plugin is available in machine
  without an Nvidia GPU, it won't be listed.

# v0.2.1 Improved Donate handling; Added DynamicSlice and DynamicSliceUpdate.

* Execute.NonDonatable -> Execute.DonateNone
* Added Execute.SetDonate
* Use `github.com/dmarkham/enumer` instead of the usual `stringer` for dtypes.
* Fixed double free of C.XlaOp pointers for Identity ops.
* Added `DynamicSlice` and `DynamicSliceUpdate`.
* Added check for matching DTypes for the common ops taking 2 operands.

# v0.2.0 GoMLX integration fixes -- GoMLX more extensive tests caught several small issues in Gopjrt.

* Moved some `dtypes` support functionality from GoMLX to Gopjrt. 
* Added BFloat16 alias.
* Renamed `FromGoType` to `FromGenericsType` and `FromType` to `FromGoType`, to maintain naming consistency.
* Added DType.Memory as an alias to DType.Size.
* Client creation immediately caches addressable devices.
* `Client.AddressableDevices` returns cached value, no errors returned.
* Added `BufferFromHost.ToDeviceNum` to allow specification of the device by device number in the addressable devices list. 
* Added `LoadedExecutable.Execute.OnDeviceNum` to allow  specification of the device by device number in the addressable devices list.
* Removed the awkward `pjrt.FlatDataToRawWithDimensions` and added the more ergonomic `Client.BufferFromHost.FromFlatDataWithDimensions`.
* Added `Buffer.ToFlatDataAndDimensions`
* Store client link with Buffer. Added `Buffer.Client` method.
* Added `Buffer.Device` and `Client.NumForDevice`.
* Properly setting client options for `pjrt.NewClient`. Added test for reading/writing `C.PJRT_NamedValues`.
* Added `xlabuilder.Shape.Memory` and `xlabuilder.NewArrayLiteralFromAny`.
* Added `xlabuilder.Op.Builder()`
* Added comments support to op_types.txt and added comments to several of the operations.
* Renamed `xlabuilder.BatchNorm{Inference,Training}` to `xlabuilder.BatchNormFor{Inference,Training}` 
* Fixed `NewArrayLiteralFromAny` to also accept scalar values, if dimensions is empty.
* Fixed `ReduceWindow` default values and allow setting values to nil.
* Fixed `Pad` to allow missing configuration for axis, per documentation.
* Fixed `ConvertDType` to convert the dtypes to the XLA version `PrimitiveType` before using.

# v0.1.2 SuppressAbseilLoggingHack

* Improved SuppressAbseilLoggingHack to supress only during the execution of a function.

# v0.1.1 New While op

* Added `While` op.
* Improved Mandelbrot example.

# v0.0.1 Initial Release

* `xlabuilder` with good coverage: all ops used by [GoMLX](github.com/gomlx/gomlx).
* `pjrt` with enough functionality coverage for [GoMLX](github.com/gomlx/gomlx) and to execute some Jax functions.
* Documentation for API, examples, one notebook (Mandelbrot) and installation details for CUDA.
* Prebuilt cpu pjrt plugin and C/C++ XlaBuilder libraries for `linux/x86-64`.
