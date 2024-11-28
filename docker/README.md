# Docker

This directory includes `Dockerfile` for different C/C++ build setups. This is mostly motivated because
of incompatibility across `glibc` and other standard C/C++ libraries across linux installations.

Notice Go doesn't suffer from this because of the static builds ... but since this depends on C/C++ (for XLA),
there is the need to have multiple builds.