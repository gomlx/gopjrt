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
func BenchmarkClient_CGO(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = must1(client.Devices())
	}
}

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
