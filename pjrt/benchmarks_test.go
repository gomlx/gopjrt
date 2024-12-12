package pjrt

import (
	"fmt"
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

// BenchmarkCGO tests a minimal CGO call.
//
// Results on cpu:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkCGO-24           8876160               129.3 ns/op
func BenchmarkCGO(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dummyCGO(unsafe.Pointer(plugin.api))
	}
}

// Benchmark tests arena
//
// Results on cpu:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkClient_CGO-24           8876160               129.3 ns/op
func BenchmarkArena(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)
	b.ResetTimer()

	numAllocationsList := []int{1, 5, 10, 100}
	allocations := make([]*int, 100)
	for _, allocType := range []string{"arena", "arenaPool", "malloc", "go+pinner"} {
		for _, numAllocations := range numAllocationsList {
			b.Run(fmt.Sprintf("%s/%d", allocType, numAllocations), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					switch allocType {
					case "arena":
						arena := newArena(1024)
						for idx := range numAllocations {
							allocations[idx] = arenaAlloc[int](arena)
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						arena.Free()
					case "arenaPool":
						arena := getArenaFromPool()
						for idx := range numAllocations {
							allocations[idx] = arenaAlloc[int](arena)
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						returnArenaToPool(arena)
					case "malloc":
						for idx := range numAllocations {
							allocations[idx] = cMalloc[int]()
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						for idx := range numAllocations {
							cFree(allocations[idx])
						}
					case "go+pinner":
						var pinner runtime.Pinner
						for idx := range numAllocations {
							v := idx
							allocations[idx] = &v
							pinner.Pin(allocations[idx])
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						pinner.Unpin()
					}
				}
			})
		}
	}
}

// BenchmarkClient_BufferFromHost benchmarks transfer time from host to device.
//
// Example command to run it:
//
//	go test . -bench=Host -run=NONE -count=20 > /tmp/benchs.txt && benchstat /tmp/benchs.txt
//
// Results on CPU:
//
//	Client_BufferFromHost/(Float32)[1_1]-24             1.206µ ± 2%
//	Client_BufferFromHost/(Float32)[10_10]-24           1.233µ ± 2%
//	Client_BufferFromHost/(Float32)[100_100]-24         2.042µ ± 2%
//	Client_BufferFromHost/(Float32)[1000_1000]-24       134.9µ ± 1%
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
// Example command to run it:
//
//	go test . -bench=Host -run=NONE -count=20 > /tmp/benchs.txt && benchstat /tmp/benchs.txt
//
// Results on CPU:
//
//	Client_BufferToHost/(Float32)[1_1]-24               2.016µ ± 4%
//	Client_BufferToHost/(Float32)[10_10]-24             2.017µ ± 4%
//	Client_BufferToHost/(Float32)[100_100]-24           5.341µ ± 1%
//	Client_BufferToHost/(Float32)[1000_1000]-24         134.0µ ± 0%
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

// BenchmarkAdd1Execution benchmarks the execution time for a minimal program.
//
// Runtimes for cpu:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1Execution/(Float32)[1_1]-24                  606991              1718 ns/op
//	BenchmarkAdd1Execution/(Float32)[10_10]-24                721738              1681 ns/op
//	BenchmarkAdd1Execution/(Float32)[100_100]-24              356961              3376 ns/op
//	BenchmarkAdd1Execution/(Float32)[1000_1000]-24             28839             41270 ns/op
func BenchmarkAdd1Execution(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
		buffers[shapeIdx] = must1(ArrayToBuffer(client, inputData[shapeIdx], s.Dimensions...))

		builder := xlabuilder.New(fmt.Sprintf("Add1/%s", s))
		// f(x) = x + 1
		x := must1(xlabuilder.Parameter(builder, "x", 0, s))
		one := must1(xlabuilder.ScalarOne(builder, s.DType))
		add1 := must1(xlabuilder.Add(x, one))
		comp := must1(builder.Build(add1))
		execs[shapeIdx] = must1(client.Compile().WithComputation(comp).Done())
	}
	defer func() {
		// Clean up -- and don't wait for the GC.
		for shapeIdx := range numShapes {
			must(buffers[shapeIdx].Destroy())
			must(execs[shapeIdx].Destroy())
		}
	}()

	// Run test for each shape
	benchShape := func(shapeIdx int) {
		buf := buffers[shapeIdx]
		exec := execs[shapeIdx]
		outputs := must1(exec.Execute(buf).Done())
		for _, output := range outputs {
			must(output.Destroy())
		}
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

// BenchmarkAdd1Div2 benchmarks the execution time for f(x) = (x+1)/2.
//
// Runtimes for cpu:
//
//	BenchmarkAdd1Div2Execution/(Float32)[1_1]-24              649802              1808 ns/op
//	BenchmarkAdd1Div2Execution/(Float32)[10_10]-24            670908              1780 ns/op
//	BenchmarkAdd1Div2Execution/(Float32)[100_100]-24          343732              3347 ns/op
//	BenchmarkAdd1Div2Execution/(Float32)[1000_1000]-24                 26298             43994 ns/op
func BenchmarkAdd1Div2Execution(b *testing.B) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
		buffers[shapeIdx] = must1(ArrayToBuffer(client, inputData[shapeIdx], s.Dimensions...))

		builder := xlabuilder.New(fmt.Sprintf("Add1/%s", s))
		// f(x) = x + 1
		x := must1(xlabuilder.Parameter(builder, "x", 0, s))
		one := must1(xlabuilder.ScalarOne(builder, s.DType))
		add1 := must1(xlabuilder.Add(x, one))
		half := must1(xlabuilder.Constant(builder, xlabuilder.NewScalarLiteral(float32(0.5))))
		div2 := must1(xlabuilder.Mul(add1, half))
		comp := must1(builder.Build(div2))
		execs[shapeIdx] = must1(client.Compile().WithComputation(comp).Done())
	}
	defer func() {
		// Clean up -- and don't wait for the GC.
		for shapeIdx := range numShapes {
			must(buffers[shapeIdx].Destroy())
			must(execs[shapeIdx].Destroy())
		}
	}()

	// Run test for each shape
	benchShape := func(shapeIdx int) {
		buf := buffers[shapeIdx]
		exec := execs[shapeIdx]
		outputs := must1(exec.Execute(buf).Done())
		for _, output := range outputs {
			must(output.Destroy())
		}
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
