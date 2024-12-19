/* DO NOT EDIT: this is a copy from chelper.go file */

package pjrt

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"reflect"
	"unsafe"
)

// File implements several CGO helper utilities.
//
// This file is copied to all package directories that need them -- because C types cannot be exported,
// See issue https://github.com/golang/go/issues/13467 .

// cFree calls C.free() on the unsafe.Pointer version of data.
func cFree[T any](data *T) {
	C.free(unsafe.Pointer(data))
}

// cSizeOf returns the size of the given type in bytes. Notice some structures may be padded, and this will
// include that space.
func cSizeOf[T any]() C.size_t {
	var ptr *T
	return C.size_t(reflect.TypeOf(ptr).Elem().Size())
}

// cMalloc allocates a T in the C heap and initializes it to zero.
// It must be manually freed with cFree() by the user.
func cMalloc[T any]() (ptr *T) {
	size := cSizeOf[T]()
	cPtr := (*T)(C.calloc(1, size))
	return cPtr
}

// cMallocArray allocates space to hold n copies of T in the C heap and initializes it to zero.
// It must be manually freed with C.free() by the user.
func cMallocArray[T any](n int) (ptr *T) {
	size := cSizeOf[T]()
	cPtr := (*T)(C.calloc(C.size_t(n), size))
	return cPtr
}

// cMallocArrayFromSlice allocates space to hold n copies of T in the C heap and copy over the slice.
// It must be manually freed with C.free() by the user.
func cMallocArrayFromSlice[T any](values []T) (ptr *T) {
	ptr = cMallocArray[T](len(values))
	dst := unsafe.Slice(ptr, len(values))
	copy(dst, values)
	return ptr
}

// cMallocArrayAndSet allocates space to hold n copies of T in the C heap, and set each element `i` with the result of
// `setFn(i)`.
// It must be manually freed with C.free() by the user.
func cMallocArrayAndSet[T any](n int, setFn func(i int) T) (ptr *T) {
	ptr = cMallocArray[T](n)
	slice := unsafe.Slice(ptr, n)
	for ii := 0; ii < n; ii++ {
		slice[ii] = setFn(ii)
	}
	return ptr
}

// cDataToSlice converts a C pointer to C allocated array of type T with count elements and return an unsafe
// slice to the data.
func cDataToSlice[T any](data unsafe.Pointer, count int) (result []T) {
	return unsafe.Slice((*T)(data), count)
}

// cCharArray converts C string given as a char pointer and length to a Go string.
// The pointer is not de-allocated.
func cCharArray(data *C.char, len C.size_t) string {
	buf := cDataToSlice[byte](unsafe.Pointer(data), int(len))
	return string(buf)
}

// cStrFree converts the allocated C string (char *) to a Go `string` and
// frees the C string immediately.
func cStrFree(cstr *C.char) (str string) {
	if cstr == nil {
		return ""
	}
	str = C.GoString(cstr)
	C.free(unsafe.Pointer(cstr))
	return
}
