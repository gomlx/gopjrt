// Package pjrt implements a Go wrapper for the PJRT_C_API.
package pjrt

// Generate automatic C-to-Go boilerplate code for pjrt_c_api.h.
//go:generate go run ../cmd/pjrt_codegen < pjrt_c_api.h

// Since CGO C types cannot cross boundaries of a package (see issue https://github.com/golang/go/issues/13467)
// We make a copy of chelper.go for every sub-directory that needs it.
//go:generate go run ../cmd/copy_go_code --original=chelper.go
