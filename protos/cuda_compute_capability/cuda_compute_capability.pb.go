// Copyright 2025 The OpenXLA Authors.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.36.4
// 	protoc        v3.21.12
// source: xla/stream_executor/cuda/cuda_compute_capability.proto

package cuda_compute_capability

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
	unsafe "unsafe"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type CudaComputeCapabilityProto struct {
	state         protoimpl.MessageState `protogen:"open.v1"`
	Major         int32                  `protobuf:"varint,1,opt,name=major,proto3" json:"major,omitempty"`
	Minor         int32                  `protobuf:"varint,2,opt,name=minor,proto3" json:"minor,omitempty"`
	unknownFields protoimpl.UnknownFields
	sizeCache     protoimpl.SizeCache
}

func (x *CudaComputeCapabilityProto) Reset() {
	*x = CudaComputeCapabilityProto{}
	mi := &file_xla_stream_executor_cuda_cuda_compute_capability_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CudaComputeCapabilityProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CudaComputeCapabilityProto) ProtoMessage() {}

func (x *CudaComputeCapabilityProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_cuda_cuda_compute_capability_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CudaComputeCapabilityProto.ProtoReflect.Descriptor instead.
func (*CudaComputeCapabilityProto) Descriptor() ([]byte, []int) {
	return file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescGZIP(), []int{0}
}

func (x *CudaComputeCapabilityProto) GetMajor() int32 {
	if x != nil {
		return x.Major
	}
	return 0
}

func (x *CudaComputeCapabilityProto) GetMinor() int32 {
	if x != nil {
		return x.Minor
	}
	return 0
}

var File_xla_stream_executor_cuda_cuda_compute_capability_proto protoreflect.FileDescriptor

var file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDesc = string([]byte{
	0x0a, 0x36, 0x78, 0x6c, 0x61, 0x2f, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65,
	0x63, 0x75, 0x74, 0x6f, 0x72, 0x2f, 0x63, 0x75, 0x64, 0x61, 0x2f, 0x63, 0x75, 0x64, 0x61, 0x5f,
	0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x5f, 0x63, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69,
	0x74, 0x79, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0f, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d,
	0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x22, 0x48, 0x0a, 0x1a, 0x43, 0x75, 0x64,
	0x61, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69,
	0x74, 0x79, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x14, 0x0a, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72, 0x12, 0x14, 0x0a,
	0x05, 0x6d, 0x69, 0x6e, 0x6f, 0x72, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6d, 0x69,
	0x6e, 0x6f, 0x72, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
})

var (
	file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescOnce sync.Once
	file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescData []byte
)

func file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescGZIP() []byte {
	file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescOnce.Do(func() {
		file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescData = protoimpl.X.CompressGZIP(unsafe.Slice(unsafe.StringData(file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDesc), len(file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDesc)))
	})
	return file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDescData
}

var file_xla_stream_executor_cuda_cuda_compute_capability_proto_msgTypes = make([]protoimpl.MessageInfo, 1)
var file_xla_stream_executor_cuda_cuda_compute_capability_proto_goTypes = []any{
	(*CudaComputeCapabilityProto)(nil), // 0: stream_executor.CudaComputeCapabilityProto
}
var file_xla_stream_executor_cuda_cuda_compute_capability_proto_depIdxs = []int32{
	0, // [0:0] is the sub-list for method output_type
	0, // [0:0] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_xla_stream_executor_cuda_cuda_compute_capability_proto_init() }
func file_xla_stream_executor_cuda_cuda_compute_capability_proto_init() {
	if File_xla_stream_executor_cuda_cuda_compute_capability_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: unsafe.Slice(unsafe.StringData(file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDesc), len(file_xla_stream_executor_cuda_cuda_compute_capability_proto_rawDesc)),
			NumEnums:      0,
			NumMessages:   1,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_xla_stream_executor_cuda_cuda_compute_capability_proto_goTypes,
		DependencyIndexes: file_xla_stream_executor_cuda_cuda_compute_capability_proto_depIdxs,
		MessageInfos:      file_xla_stream_executor_cuda_cuda_compute_capability_proto_msgTypes,
	}.Build()
	File_xla_stream_executor_cuda_cuda_compute_capability_proto = out.File
	file_xla_stream_executor_cuda_cuda_compute_capability_proto_goTypes = nil
	file_xla_stream_executor_cuda_cuda_compute_capability_proto_depIdxs = nil
}
