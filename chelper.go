package gopjrt

/*
#include <stdlib.h>
*/
import "C"
import "unsafe"

// File implements several CGO helper utilities

// cFree calls C.free() on the unsafe.Pointer version of data.
func cFree[T any](data *T) {
	C.free(unsafe.Pointer(data))
}
