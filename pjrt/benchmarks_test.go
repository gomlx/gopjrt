package pjrt

import (
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"runtime"
	"testing"
	"unsafe"
)

var testShapes = []xlabuilder.Shape{
	xlabuilder.MakeShape(dtypes.Float32, 1, 1),
	xlabuilder.MakeShape(dtypes.Float32, 10, 10),
	xlabuilder.MakeShape(dtypes.Float32, 100, 100),
	xlabuilder.MakeShape(dtypes.Float32, 1000, 1000),
}

// BenchmarkClient_CGO tests a minimal CGO call.
//
// Results on cpu:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkClient_CGO-24           8876160               129.3 ns/op
func BenchmarkClient_CGO(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = must1(pjrtClientPlatformVersion(plugin, client))
	}
}

// BenchmarkClient_BufferFromHost benchmarks transfer time from host to device.
//
// Results for cpu:
//
//	BenchmarkClient_BufferFromHost/(Float32)[1_1]-24                 1000000              1364 ns/op
//	BenchmarkClient_BufferFromHost/(Float32)[10_10]-24                820569              1413 ns/op
//	BenchmarkClient_BufferFromHost/(Float32)[100_100]-24              486066              2276 ns/op
//	BenchmarkClient_BufferFromHost/(Float32)[1000_1000]-24              7587            133828 ns/op
func BenchmarkClient_BufferFromHost(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	numShapes := len(testShapes)
	inputData := make([][]float32, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		// Set input to value of v.
		x := inputData[shapeIdx]
		s := testShapes[shapeIdx]
		buf := must1(ArrayToBuffer(client, x, s.Dimensions...))
		must(buf.Destroy())
	}

	// Warmup for each shape.
	for shapeIdx := range testShapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range testShapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

// BenchmarkClient_BufferToHost benchmarks time to transfer data from device buffer to host.
//
// Run times on cpu:
//
//	BenchmarkClient_BufferToHost/(Float32)[1_1]-24                    493348              2208 ns/op
//	BenchmarkClient_BufferToHost/(Float32)[10_10]-24                  487226              2251 ns/op
//	BenchmarkClient_BufferToHost/(Float32)[100_100]-24                216619              5580 ns/op
//	BenchmarkClient_BufferToHost/(Float32)[1000_1000]-24                9078            132018 ns/op
func BenchmarkClient_BufferToHost(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	numShapes := len(testShapes)
	// Prepare input data and upload buffers to GPU
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
		buffers[shapeIdx] = must1(ArrayToBuffer(client, inputData[shapeIdx], s.Dimensions...))
	}
	defer func() {
		for _, buf := range buffers {
			must(buf.Destroy())
		}
	}()

	// Run test for each shape
	benchShape := func(shapeIdx int) {
		buf := buffers[shapeIdx]
		rawData := unsafe.Slice((*byte)(unsafe.Pointer(&inputData[shapeIdx][0])), len(inputData[shapeIdx])*int(unsafe.Sizeof(inputData[shapeIdx][0])))
		must(buf.ToHost(rawData))
	}

	// Warmup for each shape
	for shapeIdx := range testShapes {
		for i := 0; i < 10; i++ {
			benchShape(shapeIdx)
		}
	}

	// Reset timer and start the actual benchmark
	b.ResetTimer()

	// Test each shape
	for shapeIdx, s := range testShapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(shapeIdx)
			}
		})
	}
}
