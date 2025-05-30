package pjrt

// To run benchmarks: (and fix to P-Cores if running on a Intel i9-12900K)
//	go test -c . && taskset 0xFF ./pjrt.test -test.v -test.run=Bench -test.count=1
//
// See results in https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=1369069161#gid=1369069161
import (
	"flag"
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	benchmarks "github.com/janpfeifer/go-benchmarks"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

var (
	flagBenchDuration = flag.Duration("bench_duration", 1*time.Second, "Benchmark duration")

	// testShapes used during benchmarks executing small computation graphs.
	testShapes = []xlabuilder.Shape{
		xlabuilder.MakeShape(dtypes.Float32, 1, 1),
		xlabuilder.MakeShape(dtypes.Float32, 10, 10),
		xlabuilder.MakeShape(dtypes.Float32, 100, 100),
		xlabuilder.MakeShape(dtypes.Float32, 1000, 1000),
	}
)

// TestBenchCGO benchmarks a minimal CGO call.
func TestBenchCGO(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	const repeats = 1000
	repeatedCGO := func() {
		for _ = range repeats {
			dummyCGO(unsafe.Pointer(plugin.api))
		}
	}
	benchmarks.New(benchmarks.NamedFunction{"CGOCall", repeatedCGO}).
		WithInnerRepeats(repeats).
		Done()
}

// Benchmark tests different methods to create temporary pointers to be passed to CGO.
func TestBenchArena(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	numAllocationsList := []int{1, 5, 10, 100}
	allocations := make([]*int, 100)
	testFns := make([]benchmarks.NamedFunction, 4*len(numAllocationsList))
	const repeats = 10
	idxFn := 0
	for _, allocType := range []string{"arena", "arenaPool", "malloc", "go+pinner"} {
		for _, numAllocations := range numAllocationsList {
			testFns[idxFn].Name = fmt.Sprintf("%s/%s/%d", t.Name(), allocType, numAllocations)
			var fn func()
			switch allocType {
			case "arena":
				fn = func() {
					for _ = range repeats {
						arena := newArena(1024)
						for idx := range numAllocations {
							allocations[idx] = arenaAlloc[int](arena)
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						arena.Free()
					}
				}
			case "arenaPool":
				fn = func() {
					for _ = range repeats {
						arena := plugin.getArenaFromPool()
						for idx := range numAllocations {
							allocations[idx] = arenaAlloc[int](arena)
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						plugin.returnArenaToPool(arena)
					}
				}
			case "malloc":
				fn = func() {
					for _ = range repeats {
						for idx := range numAllocations {
							allocations[idx] = cMalloc[int]()
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						for idx := range numAllocations {
							cFree(allocations[idx])
						}
					}
				}
			case "go+pinner":
				fn = func() {
					for _ = range repeats {
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
			}
			testFns[idxFn].Func = fn
			idxFn++
		}
	}
	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		WithWarmUps(10).
		Done()
}

// TestBenchBufferFromHost benchmarks host->buffer transfer time.
func TestBenchBufferFromHost(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	const repeats = 10
	numShapes := len(testShapes)
	inputData := make([][]float32, numShapes)
	testFns := make([]benchmarks.NamedFunction, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
		testFns[shapeIdx].Name = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx].Func = func() {
			for _ = range repeats {
				x := inputData[shapeIdx]
				s := testShapes[shapeIdx]
				buf := must1(ArrayToBuffer(client, x, s.Dimensions...))
				must(buf.Destroy())
			}
		}
	}
	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		WithDuration(*flagBenchDuration).
		Done()
}

// TestBenchBufferToHost benchmarks time to transfer data from device buffer to host.
//
// Results on CPU:
//
//	Benchmarks:                                                   Median         5%-tile        99%-tile
//	TestBenchBufferToHost/shape=(Float32)[1 1]                   1.684µs         1.555µs         3.762µs
//	TestBenchBufferToHost/shape=(Float32)[10 10]                 1.651µs         1.534µs         3.699µs
//	TestBenchBufferToHost/shape=(Float32)[100 100]               5.393µs         5.002µs         7.271µs
//	TestBenchBufferToHost/shape=(Float32)[1000 1000]           131.826µs       131.498µs       139.316µs
func TestBenchBufferToHost(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	const repeats = 10
	numShapes := len(testShapes)
	testFns := make([]benchmarks.NamedFunction, numShapes)
	// Prepare output data (host destination array) and upload buffers to GPU
	outputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	for shapeIdx, s := range testShapes {
		outputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			outputData[shapeIdx][i] = float32(i)
		}
		buffers[shapeIdx] = must1(ArrayToBuffer(client, outputData[shapeIdx], s.Dimensions...))
		testFns[shapeIdx].Name = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx].Func = func() {
			for _ = range repeats {
				buf := buffers[shapeIdx]
				rawData := unsafe.Slice((*byte)(unsafe.Pointer(&outputData[shapeIdx][0])), len(outputData[shapeIdx])*int(unsafe.Sizeof(outputData[shapeIdx][0])))
				must(buf.ToHost(rawData))
			}
		}
	}
	defer func() {
		for _, buf := range buffers {
			must(buf.Destroy())
		}
	}()

	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		Done()
}

// BenchmarkAdd1Execution benchmarks the execution time for a minimal program.
func TestBenchAdd1Execution(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	const repeats = 10
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	testFns := make([]benchmarks.NamedFunction, numShapes)
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
		testFns[shapeIdx].Name = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx].Func = func() {
			for _ = range repeats {
				buf := buffers[shapeIdx]
				exec := execs[shapeIdx]
				output := must1(exec.Execute(buf).Done())[0]
				must(output.Destroy())
			}
		}
	}
	defer func() {
		// Clean up -- and don't wait for the GC.
		for shapeIdx := range numShapes {
			must(buffers[shapeIdx].Destroy())
			must(execs[shapeIdx].Destroy())
		}
	}()

	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		WithWarmUps(100).
		WithDuration(*flagBenchDuration).
		Done()
}

// TestBenchAdd1Div2Execution benchmarks the execution time for f(x) = (x+1)/2.
//
// Runtimes for cpu:
//
//	Benchmarks:                                                   Median         5%-tile        99%-tile
//	TestBenchAdd1Div2Execution/shape=(Float32)[1 1]              1.536µs         1.374µs         3.522µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[10 10]            1.536µs         1.333µs         3.449µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[100 100]          2.973µs         2.638µs         5.282µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[1000 1000]       38.513µs        36.434µs        86.827µs
func TestBenchAdd1Div2Execution(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	const repeats = 10
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	testFns := make([]benchmarks.NamedFunction, numShapes)
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
		testFns[shapeIdx].Name = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx].Func = func() {
			for _ = range repeats {
				buf := buffers[shapeIdx]
				exec := execs[shapeIdx]
				output := must1(exec.Execute(buf).Done())[0]
				_ = output.Destroy()
			}
		}
	}
	defer func() {
		// Clean up -- and don't wait for the GC.
		for shapeIdx := range numShapes {
			must(buffers[shapeIdx].Destroy())
			must(execs[shapeIdx].Destroy())
		}
	}()

	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		WithWarmUps(100).
		WithDuration(*flagBenchDuration).
		Done()
}

// TestBenchAdd1Div2Execution benchmarks the execution time for f(x) = (x+1)/2.
//
// Runtimes for cpu:
//
//	Benchmarks:                                                   Median         5%-tile        99%-tile
//	TestBenchAdd1Div2Execution/shape=(Float32)[1 1]              1.536µs         1.374µs         3.522µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[10 10]            1.536µs         1.333µs         3.449µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[100 100]          2.973µs         2.638µs         5.282µs
//	TestBenchAdd1Div2Execution/shape=(Float32)[1000 1000]       38.513µs        36.434µs        86.827µs
func TestBenchMeanNormalizedExecution(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	const repeats = 10
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	testFns := make([]benchmarks.NamedFunction, numShapes)
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
		mean := must1(xlabuilder.ReduceSum(div2))
		normalized := must1(xlabuilder.Sub(div2, mean))

		comp := must1(builder.Build(normalized))
		execs[shapeIdx] = must1(client.Compile().WithComputation(comp).Done())
		testFns[shapeIdx].Name = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx].Func = func() {
			for _ = range repeats {
				buf := buffers[shapeIdx]
				exec := execs[shapeIdx]
				output := must1(exec.Execute(buf).Done())[0]
				_ = output.Destroy()
			}
		}
	}
	defer func() {
		// Clean up -- and don't wait for the GC.
		for shapeIdx := range numShapes {
			must(buffers[shapeIdx].Destroy())
			must(execs[shapeIdx].Destroy())
		}
	}()

	benchmarks.New(testFns...).
		WithInnerRepeats(repeats).
		WithWarmUps(100).
		WithDuration(*flagBenchDuration).
		Done()
}
