package xlabuilder

/*
#include <gomlx/xlabuilder/literal.h>
*/
import "C"

// Literal defines a constant value for the graph.
type Literal struct {
	cLiteralPtr *C.XlaLiteral
}

// IsNil returns true is either l is nil, or its underlying C pointer is nil.
func (l *Literal) IsNil() bool {
	return l == nil || l.cLiteralPtr == nil
}
