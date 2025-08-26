// Package stablehlo helps build a ToStableHLO program (text format) to then be
// JIT-compiled and executed by PJRT (github.com/gomlx/gopjrt/pjrt).
//
// Among its features:
//
// - Translates an API to rendered (human-readable) ToStableHLO text.
// - Shape inference: it calculates the output shapes for operations.
// - Written purely in Go, no C/C++ external dependencies.
//
// It was written as a replacement for `gopjrt/xlabuilder` and attempts to keep
// a similar or identical interface.
//
// See ToStableHLO documentation and specifications in https://openxla.org/stablehlo/spec
package stablehlo
