package gopjrt

import "C"
import "plugin"

type Error struct {
	plugin    *plugin.Plugin
	pjrtError *C.PJRT_Error
	message   string
	code      int
}

// newError creates a new wrapped PJRT_Error error. If the given pjrtError is nil, it returns nil as well
// (so no wrapping with no cost).
func newError(plugin *plugin.Plugin, pjrtError *C.PJRT_Error) *Error {
	if pjrtError == nil {
		return nil
	}

	return &Error{pjrtError: pjrtError}
}

// String implements fmt.Stringer.
// It calls the C API to retrieve the error message.
func (e *Error) String() string {
	if e == nil || e.pjrtError == nil {
		return "nil"
	}
	return e.message
}

func (e *Error) IsNil() bool {
	return e == nil || e.pjrtError == nil
}
