# Next

* Moved some `dtypes` support functionality from GoMLX to Gopjrt. 
* Added BFloat16 alias.
* Renamed `FromGoType` to `FromGenericsType` and `FromType` to `FromGoType`, to maintain naming consistency.
* Added DType.Memory as an alias to DType.Size.
* Client creation immediately caches addressable devices.
* `Client.AddressableDevices` returns cached value, no errors returned.
* Added `BufferFromHost.ToDeviceNum` to allow specification of the device by device number in the addressable devices list. 
* Added `LoadedExecutable.Execute.OnDeviceNum` to allow  specification of the device by device number in the addressable devices list.
* Removed the awkward `pjrt.FlatDataToRawWithDimensions` and added the more ergonomic `Client.BufferFromHost.FromFlatDataWithDimensions`.

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
