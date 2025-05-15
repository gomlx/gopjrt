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

// errorCode defined on a separate file, so it will work with stringer -- it doesn't work with files using cgo.

// errorCode is used by the underlying TensorFlow/XLA libraries, in Status objects.
type errorCode int

//go:generate go tool enumer -type=errorCode errorcode.go

// Values copied from tensorflow/core/protobuf/error_codes.proto.
// TODO: convert the protos definitions to Go and use that instead.
const (
	status_OK                  errorCode = 0
	status_CANCELLED           errorCode = 1
	status_UNKNOWN             errorCode = 2
	status_INVALID_ARGUMENT    errorCode = 3
	status_DEADLINE_EXCEEDED   errorCode = 4
	status_NOT_FOUND           errorCode = 5
	status_ALREADY_EXISTS      errorCode = 6
	status_PERMISSION_DENIED   errorCode = 7
	status_UNAUTHENTICATED     errorCode = 16
	status_RESOURCE_EXHAUSTED  errorCode = 8
	status_FAILED_PRECONDITION errorCode = 9
	status_ABORTED             errorCode = 10
	status_OUT_OF_RANGE        errorCode = 11
)
