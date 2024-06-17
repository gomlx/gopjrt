package xlabuilder

// #cgo LDFLAGS: -lgomlx_xlabuilder
/*
#include <gomlx/xlabuilder/op.h>
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"

// Op holds information about an Op that is part of a computation being built with an XlaBuilder.
//
// Each operation (e.g: Add, Mul) will return an Op that represents both the operation itself as well as the output
// of that operation, which can be used as input of another.
//
// While the public fields can be introspected, they shouldn't be changed.
type Op struct {
	op *C.XlaOp
}
