// Copyright 2022 The OpenXLA Authors.
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
// 	protoc-gen-go v1.34.2
// 	protoc        v5.27.2
// source: xla/stream_executor/device_description.proto

package protos

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type CudaComputeCapabilityProto struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Major int32 `protobuf:"varint,1,opt,name=major,proto3" json:"major,omitempty"`
	Minor int32 `protobuf:"varint,2,opt,name=minor,proto3" json:"minor,omitempty"`
}

func (x *CudaComputeCapabilityProto) Reset() {
	*x = CudaComputeCapabilityProto{}
	if protoimpl.UnsafeEnabled {
		mi := &file_xla_stream_executor_device_description_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CudaComputeCapabilityProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CudaComputeCapabilityProto) ProtoMessage() {}

func (x *CudaComputeCapabilityProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_device_description_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
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
	return file_xla_stream_executor_device_description_proto_rawDescGZIP(), []int{0}
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

type RocmComputeCapabilityProto struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	GcnArchName string `protobuf:"bytes,1,opt,name=gcn_arch_name,json=gcnArchName,proto3" json:"gcn_arch_name,omitempty"`
}

func (x *RocmComputeCapabilityProto) Reset() {
	*x = RocmComputeCapabilityProto{}
	if protoimpl.UnsafeEnabled {
		mi := &file_xla_stream_executor_device_description_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RocmComputeCapabilityProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RocmComputeCapabilityProto) ProtoMessage() {}

func (x *RocmComputeCapabilityProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_device_description_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RocmComputeCapabilityProto.ProtoReflect.Descriptor instead.
func (*RocmComputeCapabilityProto) Descriptor() ([]byte, []int) {
	return file_xla_stream_executor_device_description_proto_rawDescGZIP(), []int{1}
}

func (x *RocmComputeCapabilityProto) GetGcnArchName() string {
	if x != nil {
		return x.GcnArchName
	}
	return ""
}

type GpuDeviceInfoProto struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ThreadsPerBlockLimit      int32   `protobuf:"varint,1,opt,name=threads_per_block_limit,json=threadsPerBlockLimit,proto3" json:"threads_per_block_limit,omitempty"`
	ThreadsPerWarp            int32   `protobuf:"varint,2,opt,name=threads_per_warp,json=threadsPerWarp,proto3" json:"threads_per_warp,omitempty"`
	SharedMemoryPerBlock      int32   `protobuf:"varint,3,opt,name=shared_memory_per_block,json=sharedMemoryPerBlock,proto3" json:"shared_memory_per_block,omitempty"`
	SharedMemoryPerCore       int32   `protobuf:"varint,4,opt,name=shared_memory_per_core,json=sharedMemoryPerCore,proto3" json:"shared_memory_per_core,omitempty"`
	ThreadsPerCoreLimit       int32   `protobuf:"varint,5,opt,name=threads_per_core_limit,json=threadsPerCoreLimit,proto3" json:"threads_per_core_limit,omitempty"`
	CoreCount                 int32   `protobuf:"varint,6,opt,name=core_count,json=coreCount,proto3" json:"core_count,omitempty"`
	FpusPerCore               int64   `protobuf:"varint,7,opt,name=fpus_per_core,json=fpusPerCore,proto3" json:"fpus_per_core,omitempty"`
	BlockDimLimitX            int32   `protobuf:"varint,8,opt,name=block_dim_limit_x,json=blockDimLimitX,proto3" json:"block_dim_limit_x,omitempty"`
	BlockDimLimitY            int32   `protobuf:"varint,9,opt,name=block_dim_limit_y,json=blockDimLimitY,proto3" json:"block_dim_limit_y,omitempty"`
	BlockDimLimitZ            int32   `protobuf:"varint,10,opt,name=block_dim_limit_z,json=blockDimLimitZ,proto3" json:"block_dim_limit_z,omitempty"`
	MemoryBandwidth           int64   `protobuf:"varint,11,opt,name=memory_bandwidth,json=memoryBandwidth,proto3" json:"memory_bandwidth,omitempty"`
	L2CacheSize               int64   `protobuf:"varint,12,opt,name=l2_cache_size,json=l2CacheSize,proto3" json:"l2_cache_size,omitempty"`
	ClockRateGhz              float32 `protobuf:"fixed32,13,opt,name=clock_rate_ghz,json=clockRateGhz,proto3" json:"clock_rate_ghz,omitempty"`
	DeviceMemorySize          int64   `protobuf:"varint,14,opt,name=device_memory_size,json=deviceMemorySize,proto3" json:"device_memory_size,omitempty"`
	SharedMemoryPerBlockOptin int32   `protobuf:"varint,15,opt,name=shared_memory_per_block_optin,json=sharedMemoryPerBlockOptin,proto3" json:"shared_memory_per_block_optin,omitempty"`
	// Types that are assignable to ComputeCapability:
	//
	//	*GpuDeviceInfoProto_CudaComputeCapability
	//	*GpuDeviceInfoProto_RocmComputeCapability
	ComputeCapability isGpuDeviceInfoProto_ComputeCapability `protobuf_oneof:"compute_capability"`
}

func (x *GpuDeviceInfoProto) Reset() {
	*x = GpuDeviceInfoProto{}
	if protoimpl.UnsafeEnabled {
		mi := &file_xla_stream_executor_device_description_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GpuDeviceInfoProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GpuDeviceInfoProto) ProtoMessage() {}

func (x *GpuDeviceInfoProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_device_description_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GpuDeviceInfoProto.ProtoReflect.Descriptor instead.
func (*GpuDeviceInfoProto) Descriptor() ([]byte, []int) {
	return file_xla_stream_executor_device_description_proto_rawDescGZIP(), []int{2}
}

func (x *GpuDeviceInfoProto) GetThreadsPerBlockLimit() int32 {
	if x != nil {
		return x.ThreadsPerBlockLimit
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetThreadsPerWarp() int32 {
	if x != nil {
		return x.ThreadsPerWarp
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetSharedMemoryPerBlock() int32 {
	if x != nil {
		return x.SharedMemoryPerBlock
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetSharedMemoryPerCore() int32 {
	if x != nil {
		return x.SharedMemoryPerCore
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetThreadsPerCoreLimit() int32 {
	if x != nil {
		return x.ThreadsPerCoreLimit
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetCoreCount() int32 {
	if x != nil {
		return x.CoreCount
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetFpusPerCore() int64 {
	if x != nil {
		return x.FpusPerCore
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetBlockDimLimitX() int32 {
	if x != nil {
		return x.BlockDimLimitX
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetBlockDimLimitY() int32 {
	if x != nil {
		return x.BlockDimLimitY
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetBlockDimLimitZ() int32 {
	if x != nil {
		return x.BlockDimLimitZ
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetMemoryBandwidth() int64 {
	if x != nil {
		return x.MemoryBandwidth
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetL2CacheSize() int64 {
	if x != nil {
		return x.L2CacheSize
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetClockRateGhz() float32 {
	if x != nil {
		return x.ClockRateGhz
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetDeviceMemorySize() int64 {
	if x != nil {
		return x.DeviceMemorySize
	}
	return 0
}

func (x *GpuDeviceInfoProto) GetSharedMemoryPerBlockOptin() int32 {
	if x != nil {
		return x.SharedMemoryPerBlockOptin
	}
	return 0
}

func (m *GpuDeviceInfoProto) GetComputeCapability() isGpuDeviceInfoProto_ComputeCapability {
	if m != nil {
		return m.ComputeCapability
	}
	return nil
}

func (x *GpuDeviceInfoProto) GetCudaComputeCapability() *CudaComputeCapabilityProto {
	if x, ok := x.GetComputeCapability().(*GpuDeviceInfoProto_CudaComputeCapability); ok {
		return x.CudaComputeCapability
	}
	return nil
}

func (x *GpuDeviceInfoProto) GetRocmComputeCapability() *RocmComputeCapabilityProto {
	if x, ok := x.GetComputeCapability().(*GpuDeviceInfoProto_RocmComputeCapability); ok {
		return x.RocmComputeCapability
	}
	return nil
}

type isGpuDeviceInfoProto_ComputeCapability interface {
	isGpuDeviceInfoProto_ComputeCapability()
}

type GpuDeviceInfoProto_CudaComputeCapability struct {
	CudaComputeCapability *CudaComputeCapabilityProto `protobuf:"bytes,16,opt,name=cuda_compute_capability,json=cudaComputeCapability,proto3,oneof"`
}

type GpuDeviceInfoProto_RocmComputeCapability struct {
	RocmComputeCapability *RocmComputeCapabilityProto `protobuf:"bytes,17,opt,name=rocm_compute_capability,json=rocmComputeCapability,proto3,oneof"`
}

func (*GpuDeviceInfoProto_CudaComputeCapability) isGpuDeviceInfoProto_ComputeCapability() {}

func (*GpuDeviceInfoProto_RocmComputeCapability) isGpuDeviceInfoProto_ComputeCapability() {}

type DnnVersionInfoProto struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Major int32 `protobuf:"varint,1,opt,name=major,proto3" json:"major,omitempty"`
	Minor int32 `protobuf:"varint,2,opt,name=minor,proto3" json:"minor,omitempty"`
	Patch int32 `protobuf:"varint,3,opt,name=patch,proto3" json:"patch,omitempty"`
}

func (x *DnnVersionInfoProto) Reset() {
	*x = DnnVersionInfoProto{}
	if protoimpl.UnsafeEnabled {
		mi := &file_xla_stream_executor_device_description_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DnnVersionInfoProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DnnVersionInfoProto) ProtoMessage() {}

func (x *DnnVersionInfoProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_device_description_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DnnVersionInfoProto.ProtoReflect.Descriptor instead.
func (*DnnVersionInfoProto) Descriptor() ([]byte, []int) {
	return file_xla_stream_executor_device_description_proto_rawDescGZIP(), []int{3}
}

func (x *DnnVersionInfoProto) GetMajor() int32 {
	if x != nil {
		return x.Major
	}
	return 0
}

func (x *DnnVersionInfoProto) GetMinor() int32 {
	if x != nil {
		return x.Minor
	}
	return 0
}

func (x *DnnVersionInfoProto) GetPatch() int32 {
	if x != nil {
		return x.Patch
	}
	return 0
}

type GpuTargetConfigProto struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	GpuDeviceInfo  *GpuDeviceInfoProto  `protobuf:"bytes,1,opt,name=gpu_device_info,json=gpuDeviceInfo,proto3" json:"gpu_device_info,omitempty"`
	PlatformName   string               `protobuf:"bytes,4,opt,name=platform_name,json=platformName,proto3" json:"platform_name,omitempty"`
	DnnVersionInfo *DnnVersionInfoProto `protobuf:"bytes,5,opt,name=dnn_version_info,json=dnnVersionInfo,proto3" json:"dnn_version_info,omitempty"`
	// TODO(b/248362914): Autotuning results should be separate from
	// GpuTargetConfig because autotuning can be updated regularly separate from
	// the target.
	AutotuneResults      *AutotuneResults `protobuf:"bytes,6,opt,name=autotune_results,json=autotuneResults,proto3" json:"autotune_results,omitempty"`
	DeviceDescriptionStr string           `protobuf:"bytes,7,opt,name=device_description_str,json=deviceDescriptionStr,proto3" json:"device_description_str,omitempty"`
}

func (x *GpuTargetConfigProto) Reset() {
	*x = GpuTargetConfigProto{}
	if protoimpl.UnsafeEnabled {
		mi := &file_xla_stream_executor_device_description_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GpuTargetConfigProto) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GpuTargetConfigProto) ProtoMessage() {}

func (x *GpuTargetConfigProto) ProtoReflect() protoreflect.Message {
	mi := &file_xla_stream_executor_device_description_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GpuTargetConfigProto.ProtoReflect.Descriptor instead.
func (*GpuTargetConfigProto) Descriptor() ([]byte, []int) {
	return file_xla_stream_executor_device_description_proto_rawDescGZIP(), []int{4}
}

func (x *GpuTargetConfigProto) GetGpuDeviceInfo() *GpuDeviceInfoProto {
	if x != nil {
		return x.GpuDeviceInfo
	}
	return nil
}

func (x *GpuTargetConfigProto) GetPlatformName() string {
	if x != nil {
		return x.PlatformName
	}
	return ""
}

func (x *GpuTargetConfigProto) GetDnnVersionInfo() *DnnVersionInfoProto {
	if x != nil {
		return x.DnnVersionInfo
	}
	return nil
}

func (x *GpuTargetConfigProto) GetAutotuneResults() *AutotuneResults {
	if x != nil {
		return x.AutotuneResults
	}
	return nil
}

func (x *GpuTargetConfigProto) GetDeviceDescriptionStr() string {
	if x != nil {
		return x.DeviceDescriptionStr
	}
	return ""
}

var File_xla_stream_executor_device_description_proto protoreflect.FileDescriptor

var file_xla_stream_executor_device_description_proto_rawDesc = []byte{
	0x0a, 0x2c, 0x78, 0x6c, 0x61, 0x2f, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65,
	0x63, 0x75, 0x74, 0x6f, 0x72, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x64, 0x65, 0x73,
	0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0f,
	0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x1a,
	0x1a, 0x78, 0x6c, 0x61, 0x2f, 0x61, 0x75, 0x74, 0x6f, 0x74, 0x75, 0x6e, 0x65, 0x5f, 0x72, 0x65,
	0x73, 0x75, 0x6c, 0x74, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x48, 0x0a, 0x1a, 0x43,
	0x75, 0x64, 0x61, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70, 0x61, 0x62, 0x69,
	0x6c, 0x69, 0x74, 0x79, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x14, 0x0a, 0x05, 0x6d, 0x61, 0x6a,
	0x6f, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72, 0x12,
	0x14, 0x0a, 0x05, 0x6d, 0x69, 0x6e, 0x6f, 0x72, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05,
	0x6d, 0x69, 0x6e, 0x6f, 0x72, 0x22, 0x40, 0x0a, 0x1a, 0x52, 0x6f, 0x63, 0x6d, 0x43, 0x6f, 0x6d,
	0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x50, 0x72,
	0x6f, 0x74, 0x6f, 0x12, 0x22, 0x0a, 0x0d, 0x67, 0x63, 0x6e, 0x5f, 0x61, 0x72, 0x63, 0x68, 0x5f,
	0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0b, 0x67, 0x63, 0x6e, 0x41,
	0x72, 0x63, 0x68, 0x4e, 0x61, 0x6d, 0x65, 0x22, 0xa3, 0x07, 0x0a, 0x12, 0x47, 0x70, 0x75, 0x44,
	0x65, 0x76, 0x69, 0x63, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x35,
	0x0a, 0x17, 0x74, 0x68, 0x72, 0x65, 0x61, 0x64, 0x73, 0x5f, 0x70, 0x65, 0x72, 0x5f, 0x62, 0x6c,
	0x6f, 0x63, 0x6b, 0x5f, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x14, 0x74, 0x68, 0x72, 0x65, 0x61, 0x64, 0x73, 0x50, 0x65, 0x72, 0x42, 0x6c, 0x6f, 0x63, 0x6b,
	0x4c, 0x69, 0x6d, 0x69, 0x74, 0x12, 0x28, 0x0a, 0x10, 0x74, 0x68, 0x72, 0x65, 0x61, 0x64, 0x73,
	0x5f, 0x70, 0x65, 0x72, 0x5f, 0x77, 0x61, 0x72, 0x70, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x0e, 0x74, 0x68, 0x72, 0x65, 0x61, 0x64, 0x73, 0x50, 0x65, 0x72, 0x57, 0x61, 0x72, 0x70, 0x12,
	0x35, 0x0a, 0x17, 0x73, 0x68, 0x61, 0x72, 0x65, 0x64, 0x5f, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79,
	0x5f, 0x70, 0x65, 0x72, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x18, 0x03, 0x20, 0x01, 0x28, 0x05,
	0x52, 0x14, 0x73, 0x68, 0x61, 0x72, 0x65, 0x64, 0x4d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x50, 0x65,
	0x72, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x12, 0x33, 0x0a, 0x16, 0x73, 0x68, 0x61, 0x72, 0x65, 0x64,
	0x5f, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x5f, 0x70, 0x65, 0x72, 0x5f, 0x63, 0x6f, 0x72, 0x65,
	0x18, 0x04, 0x20, 0x01, 0x28, 0x05, 0x52, 0x13, 0x73, 0x68, 0x61, 0x72, 0x65, 0x64, 0x4d, 0x65,
	0x6d, 0x6f, 0x72, 0x79, 0x50, 0x65, 0x72, 0x43, 0x6f, 0x72, 0x65, 0x12, 0x33, 0x0a, 0x16, 0x74,
	0x68, 0x72, 0x65, 0x61, 0x64, 0x73, 0x5f, 0x70, 0x65, 0x72, 0x5f, 0x63, 0x6f, 0x72, 0x65, 0x5f,
	0x6c, 0x69, 0x6d, 0x69, 0x74, 0x18, 0x05, 0x20, 0x01, 0x28, 0x05, 0x52, 0x13, 0x74, 0x68, 0x72,
	0x65, 0x61, 0x64, 0x73, 0x50, 0x65, 0x72, 0x43, 0x6f, 0x72, 0x65, 0x4c, 0x69, 0x6d, 0x69, 0x74,
	0x12, 0x1d, 0x0a, 0x0a, 0x63, 0x6f, 0x72, 0x65, 0x5f, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x18, 0x06,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x09, 0x63, 0x6f, 0x72, 0x65, 0x43, 0x6f, 0x75, 0x6e, 0x74, 0x12,
	0x22, 0x0a, 0x0d, 0x66, 0x70, 0x75, 0x73, 0x5f, 0x70, 0x65, 0x72, 0x5f, 0x63, 0x6f, 0x72, 0x65,
	0x18, 0x07, 0x20, 0x01, 0x28, 0x03, 0x52, 0x0b, 0x66, 0x70, 0x75, 0x73, 0x50, 0x65, 0x72, 0x43,
	0x6f, 0x72, 0x65, 0x12, 0x29, 0x0a, 0x11, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x64, 0x69, 0x6d,
	0x5f, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x5f, 0x78, 0x18, 0x08, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0e,
	0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x44, 0x69, 0x6d, 0x4c, 0x69, 0x6d, 0x69, 0x74, 0x58, 0x12, 0x29,
	0x0a, 0x11, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x64, 0x69, 0x6d, 0x5f, 0x6c, 0x69, 0x6d, 0x69,
	0x74, 0x5f, 0x79, 0x18, 0x09, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0e, 0x62, 0x6c, 0x6f, 0x63, 0x6b,
	0x44, 0x69, 0x6d, 0x4c, 0x69, 0x6d, 0x69, 0x74, 0x59, 0x12, 0x29, 0x0a, 0x11, 0x62, 0x6c, 0x6f,
	0x63, 0x6b, 0x5f, 0x64, 0x69, 0x6d, 0x5f, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x5f, 0x7a, 0x18, 0x0a,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x0e, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x44, 0x69, 0x6d, 0x4c, 0x69,
	0x6d, 0x69, 0x74, 0x5a, 0x12, 0x29, 0x0a, 0x10, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x5f, 0x62,
	0x61, 0x6e, 0x64, 0x77, 0x69, 0x64, 0x74, 0x68, 0x18, 0x0b, 0x20, 0x01, 0x28, 0x03, 0x52, 0x0f,
	0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x42, 0x61, 0x6e, 0x64, 0x77, 0x69, 0x64, 0x74, 0x68, 0x12,
	0x22, 0x0a, 0x0d, 0x6c, 0x32, 0x5f, 0x63, 0x61, 0x63, 0x68, 0x65, 0x5f, 0x73, 0x69, 0x7a, 0x65,
	0x18, 0x0c, 0x20, 0x01, 0x28, 0x03, 0x52, 0x0b, 0x6c, 0x32, 0x43, 0x61, 0x63, 0x68, 0x65, 0x53,
	0x69, 0x7a, 0x65, 0x12, 0x24, 0x0a, 0x0e, 0x63, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x72, 0x61, 0x74,
	0x65, 0x5f, 0x67, 0x68, 0x7a, 0x18, 0x0d, 0x20, 0x01, 0x28, 0x02, 0x52, 0x0c, 0x63, 0x6c, 0x6f,
	0x63, 0x6b, 0x52, 0x61, 0x74, 0x65, 0x47, 0x68, 0x7a, 0x12, 0x2c, 0x0a, 0x12, 0x64, 0x65, 0x76,
	0x69, 0x63, 0x65, 0x5f, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x18,
	0x0e, 0x20, 0x01, 0x28, 0x03, 0x52, 0x10, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x4d, 0x65, 0x6d,
	0x6f, 0x72, 0x79, 0x53, 0x69, 0x7a, 0x65, 0x12, 0x40, 0x0a, 0x1d, 0x73, 0x68, 0x61, 0x72, 0x65,
	0x64, 0x5f, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x5f, 0x70, 0x65, 0x72, 0x5f, 0x62, 0x6c, 0x6f,
	0x63, 0x6b, 0x5f, 0x6f, 0x70, 0x74, 0x69, 0x6e, 0x18, 0x0f, 0x20, 0x01, 0x28, 0x05, 0x52, 0x19,
	0x73, 0x68, 0x61, 0x72, 0x65, 0x64, 0x4d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x50, 0x65, 0x72, 0x42,
	0x6c, 0x6f, 0x63, 0x6b, 0x4f, 0x70, 0x74, 0x69, 0x6e, 0x12, 0x65, 0x0a, 0x17, 0x63, 0x75, 0x64,
	0x61, 0x5f, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x5f, 0x63, 0x61, 0x70, 0x61, 0x62, 0x69,
	0x6c, 0x69, 0x74, 0x79, 0x18, 0x10, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x2b, 0x2e, 0x73, 0x74, 0x72,
	0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x2e, 0x43, 0x75, 0x64,
	0x61, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69,
	0x74, 0x79, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x48, 0x00, 0x52, 0x15, 0x63, 0x75, 0x64, 0x61, 0x43,
	0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79,
	0x12, 0x65, 0x0a, 0x17, 0x72, 0x6f, 0x63, 0x6d, 0x5f, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65,
	0x5f, 0x63, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x18, 0x11, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x2b, 0x2e, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65, 0x63, 0x75,
	0x74, 0x6f, 0x72, 0x2e, 0x52, 0x6f, 0x63, 0x6d, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43,
	0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x48, 0x00,
	0x52, 0x15, 0x72, 0x6f, 0x63, 0x6d, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x43, 0x61, 0x70,
	0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x42, 0x14, 0x0a, 0x12, 0x63, 0x6f, 0x6d, 0x70, 0x75,
	0x74, 0x65, 0x5f, 0x63, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x22, 0x57, 0x0a,
	0x13, 0x44, 0x6e, 0x6e, 0x56, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x49, 0x6e, 0x66, 0x6f, 0x50,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x14, 0x0a, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x05, 0x52, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72, 0x12, 0x14, 0x0a, 0x05, 0x6d, 0x69,
	0x6e, 0x6f, 0x72, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6d, 0x69, 0x6e, 0x6f, 0x72,
	0x12, 0x14, 0x0a, 0x05, 0x70, 0x61, 0x74, 0x63, 0x68, 0x18, 0x03, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x05, 0x70, 0x61, 0x74, 0x63, 0x68, 0x22, 0x8d, 0x03, 0x0a, 0x14, 0x47, 0x70, 0x75, 0x54, 0x61,
	0x72, 0x67, 0x65, 0x74, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x12,
	0x4b, 0x0a, 0x0f, 0x67, 0x70, 0x75, 0x5f, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x69, 0x6e,
	0x66, 0x6f, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x23, 0x2e, 0x73, 0x74, 0x72, 0x65, 0x61,
	0x6d, 0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x2e, 0x47, 0x70, 0x75, 0x44, 0x65,
	0x76, 0x69, 0x63, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x52, 0x0d, 0x67,
	0x70, 0x75, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x23, 0x0a, 0x0d,
	0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x04, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x0c, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x4e, 0x61, 0x6d,
	0x65, 0x12, 0x4e, 0x0a, 0x10, 0x64, 0x6e, 0x6e, 0x5f, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e,
	0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x18, 0x05, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x24, 0x2e, 0x73, 0x74,
	0x72, 0x65, 0x61, 0x6d, 0x5f, 0x65, 0x78, 0x65, 0x63, 0x75, 0x74, 0x6f, 0x72, 0x2e, 0x44, 0x6e,
	0x6e, 0x56, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x49, 0x6e, 0x66, 0x6f, 0x50, 0x72, 0x6f, 0x74,
	0x6f, 0x52, 0x0e, 0x64, 0x6e, 0x6e, 0x56, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x49, 0x6e, 0x66,
	0x6f, 0x12, 0x3f, 0x0a, 0x10, 0x61, 0x75, 0x74, 0x6f, 0x74, 0x75, 0x6e, 0x65, 0x5f, 0x72, 0x65,
	0x73, 0x75, 0x6c, 0x74, 0x73, 0x18, 0x06, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x14, 0x2e, 0x78, 0x6c,
	0x61, 0x2e, 0x41, 0x75, 0x74, 0x6f, 0x74, 0x75, 0x6e, 0x65, 0x52, 0x65, 0x73, 0x75, 0x6c, 0x74,
	0x73, 0x52, 0x0f, 0x61, 0x75, 0x74, 0x6f, 0x74, 0x75, 0x6e, 0x65, 0x52, 0x65, 0x73, 0x75, 0x6c,
	0x74, 0x73, 0x12, 0x34, 0x0a, 0x16, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x64, 0x65, 0x73,
	0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x73, 0x74, 0x72, 0x18, 0x07, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x14, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x44, 0x65, 0x73, 0x63, 0x72, 0x69,
	0x70, 0x74, 0x69, 0x6f, 0x6e, 0x53, 0x74, 0x72, 0x4a, 0x04, 0x08, 0x02, 0x10, 0x03, 0x4a, 0x04,
	0x08, 0x03, 0x10, 0x04, 0x52, 0x17, 0x63, 0x75, 0x64, 0x61, 0x5f, 0x63, 0x6f, 0x6d, 0x70, 0x75,
	0x74, 0x65, 0x5f, 0x63, 0x61, 0x70, 0x61, 0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x52, 0x17, 0x72,
	0x6f, 0x63, 0x6d, 0x5f, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x5f, 0x63, 0x61, 0x70, 0x61,
	0x62, 0x69, 0x6c, 0x69, 0x74, 0x79, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_xla_stream_executor_device_description_proto_rawDescOnce sync.Once
	file_xla_stream_executor_device_description_proto_rawDescData = file_xla_stream_executor_device_description_proto_rawDesc
)

func file_xla_stream_executor_device_description_proto_rawDescGZIP() []byte {
	file_xla_stream_executor_device_description_proto_rawDescOnce.Do(func() {
		file_xla_stream_executor_device_description_proto_rawDescData = protoimpl.X.CompressGZIP(file_xla_stream_executor_device_description_proto_rawDescData)
	})
	return file_xla_stream_executor_device_description_proto_rawDescData
}

var file_xla_stream_executor_device_description_proto_msgTypes = make([]protoimpl.MessageInfo, 5)
var file_xla_stream_executor_device_description_proto_goTypes = []any{
	(*CudaComputeCapabilityProto)(nil), // 0: stream_executor.CudaComputeCapabilityProto
	(*RocmComputeCapabilityProto)(nil), // 1: stream_executor.RocmComputeCapabilityProto
	(*GpuDeviceInfoProto)(nil),         // 2: stream_executor.GpuDeviceInfoProto
	(*DnnVersionInfoProto)(nil),        // 3: stream_executor.DnnVersionInfoProto
	(*GpuTargetConfigProto)(nil),       // 4: stream_executor.GpuTargetConfigProto
	(*AutotuneResults)(nil),            // 5: xla.AutotuneResults
}
var file_xla_stream_executor_device_description_proto_depIdxs = []int32{
	0, // 0: stream_executor.GpuDeviceInfoProto.cuda_compute_capability:type_name -> stream_executor.CudaComputeCapabilityProto
	1, // 1: stream_executor.GpuDeviceInfoProto.rocm_compute_capability:type_name -> stream_executor.RocmComputeCapabilityProto
	2, // 2: stream_executor.GpuTargetConfigProto.gpu_device_info:type_name -> stream_executor.GpuDeviceInfoProto
	3, // 3: stream_executor.GpuTargetConfigProto.dnn_version_info:type_name -> stream_executor.DnnVersionInfoProto
	5, // 4: stream_executor.GpuTargetConfigProto.autotune_results:type_name -> xla.AutotuneResults
	5, // [5:5] is the sub-list for method output_type
	5, // [5:5] is the sub-list for method input_type
	5, // [5:5] is the sub-list for extension type_name
	5, // [5:5] is the sub-list for extension extendee
	0, // [0:5] is the sub-list for field type_name
}

func init() { file_xla_stream_executor_device_description_proto_init() }
func file_xla_stream_executor_device_description_proto_init() {
	if File_xla_stream_executor_device_description_proto != nil {
		return
	}
	file_xla_autotune_results_proto_init()
	if !protoimpl.UnsafeEnabled {
		file_xla_stream_executor_device_description_proto_msgTypes[0].Exporter = func(v any, i int) any {
			switch v := v.(*CudaComputeCapabilityProto); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_xla_stream_executor_device_description_proto_msgTypes[1].Exporter = func(v any, i int) any {
			switch v := v.(*RocmComputeCapabilityProto); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_xla_stream_executor_device_description_proto_msgTypes[2].Exporter = func(v any, i int) any {
			switch v := v.(*GpuDeviceInfoProto); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_xla_stream_executor_device_description_proto_msgTypes[3].Exporter = func(v any, i int) any {
			switch v := v.(*DnnVersionInfoProto); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_xla_stream_executor_device_description_proto_msgTypes[4].Exporter = func(v any, i int) any {
			switch v := v.(*GpuTargetConfigProto); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	file_xla_stream_executor_device_description_proto_msgTypes[2].OneofWrappers = []any{
		(*GpuDeviceInfoProto_CudaComputeCapability)(nil),
		(*GpuDeviceInfoProto_RocmComputeCapability)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_xla_stream_executor_device_description_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   5,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_xla_stream_executor_device_description_proto_goTypes,
		DependencyIndexes: file_xla_stream_executor_device_description_proto_depIdxs,
		MessageInfos:      file_xla_stream_executor_device_description_proto_msgTypes,
	}.Build()
	File_xla_stream_executor_device_description_proto = out.File
	file_xla_stream_executor_device_description_proto_rawDesc = nil
	file_xla_stream_executor_device_description_proto_goTypes = nil
	file_xla_stream_executor_device_description_proto_depIdxs = nil
}