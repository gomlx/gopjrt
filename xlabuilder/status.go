/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package xlabuilder

// This file implements a wrapper over C++'s Status and StatusOr objects.

// #include <string.h>
// #include <gomlx/xlabuilder/utils.h>
import "C"
import (
	"fmt"
	"unsafe"
)

// unsafePointerOrError converts a StatusOr structure to either an unsafe.Pointer with the data
// or the Status converted to an error message and then freed.
func unsafePointerOrError(s C.StatusOr) (unsafe.Pointer, error) {
	if err := errorFromStatus(s.status); err != nil {
		return nil, err
	}
	return s.value, nil
}

// pointerOrError converts a StatusOr structure to either a pointer to T with the data
// or the Status converted to an error message and then freed.
func pointerOrError[T any](s C.StatusOr) (t *T, err error) {
	var ptr unsafe.Pointer
	ptr, err = unsafePointerOrError(s)
	if err != nil {
		return
	}
	t = (*T)(ptr)
	return
}

// errorFromStatus converts a *C.XlaStatus returned to an error or nil if there were no
// errors or if status == nil.
// It also frees the given *C.XlaStatus.
func errorFromStatus(status unsafe.Pointer) (err error) {
	if status == nil {
		return // no error
	}
	defer C.XlaStatusFree(status)
	code := errorCode(C.XlaStatusCode(status))
	if code == status_OK {
		return
	}
	msg := cStrFree(C.XlaStatusErrorMessage(status))
	err = fmt.Errorf("%d: %s", code, msg)
	return
}
