// Package pjrt implements a Go wrapper for the PJRT_C_API.
package pjrt

import "github.com/pkg/errors"

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../internal/cmd/pjrt_codegen

// Since CGO C types cannot cross boundaries of a package (see issue https://github.com/golang/go/issues/13467)
// We make a copy of chelper.go for every sub-directory that needs it.
//go:generate go run ../internal/cmd/copy_go_code --original=internal/chelper.go

// panicf panics with a formatted description.
//
// It is only used for "bugs in the code" -- when parameters don't follow the specifications.
// In principle, it should never happen -- the same way nil-pointer panics should never happen.
func panicf(format string, args ...any) {
	panic(errors.Errorf(format, args...))
}
