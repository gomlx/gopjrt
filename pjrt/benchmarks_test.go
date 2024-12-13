package pjrt

// To run benchmarks:
//	go test . -test.v -test.run=Bench -test.count=1
//
// It is configured to report the median, the 5%-tile and the 99%-tile.
// All metrics recorded on a 12th Gen Intel(R) Core(TM) i9-12900K.

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/streadway/quantile"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
	"unsafe"
)

var testShapes = []xlabuilder.Shape{
	xlabuilder.MakeShape(dtypes.Float32, 1, 1),
	xlabuilder.MakeShape(dtypes.Float32, 10, 10),
	xlabuilder.MakeShape(dtypes.Float32, 100, 100),
	xlabuilder.MakeShape(dtypes.Float32, 1000, 1000),
}

var (
	benchmarkDuration = 1 * time.Second
)

func observeNanoseconds(est *quantile.Estimator, start time.Time) {
	est.Add(float64(time.Now().Sub(start)) / float64(time.Nanosecond))
}

func nanosecondsEstimate(est *quantile.Estimator, quantile float64) time.Duration {
	return time.Duration(int(est.Get(quantile))) * time.Nanosecond
}

func benchmarkOneFunc(fn func()) (median, percentile5, percentile99 time.Duration) {
	estimator := quantile.New(
		quantile.Known(0.50, 0.0001),
		quantile.Known(0.05, 0.0001),
		quantile.Known(0.99, 0.0001),
	)
	timer := time.NewTimer(benchmarkDuration)
collection:
	for {
		select {
		case <-timer.C:
			break collection
		default:
			start := time.Now()
			fn()
			observeNanoseconds(estimator, start)
		}
	}
	median = nanosecondsEstimate(estimator, 0.50)
	percentile5 = nanosecondsEstimate(estimator, 0.05)
	percentile99 = nanosecondsEstimate(estimator, 0.99)
	return
}

// benchmarkNamedFunctions benchmarks and pretty print results, with median and 5-percentile.
//
// Before each function is benchmarked, it is called warmUpCalls times.
//
// For very fast code, where even the extra wrapping in a function call makes a difference, you can execute
// the code. The total time reported will be divided by repeats, if it's not 0.
func benchmarkNamedFunctions(names []string, fns []func(), warmUpCalls int, repeats int) {
	header := "Benchmarks:"
	maxLen := len(header)
	runeCount := make([]int, len(names))
	for ii, name := range names {
		runeCount[ii] = utf8.RuneCountInString(name)
		maxLen = max(maxLen, runeCount[ii])
	}

	// Header
	extraSpaces := maxLen - len(header)
	if extraSpaces > 0 {
		header = header + strings.Repeat(" ", extraSpaces)
	}
	fmt.Printf("%s\t%12s\t%12s\t%12s\n", header, "Median", "5%-tile", "99%-tile")

	for ii, fn := range fns {
		// Warm-up
		for _ = range warmUpCalls {
			fn()
		}

		// Collect benchmark estimations.
		median, percentile5, percentile99 := benchmarkOneFunc(fn)
		if repeats > 1 {
			median /= time.Duration(repeats)
			percentile5 /= time.Duration(repeats)
			percentile99 /= time.Duration(repeats)
		}

		// Pretty-print.
		name := names[ii]
		extraSpaces := maxLen - runeCount[ii]
		if extraSpaces > 0 {
			name = name + strings.Repeat(" ", extraSpaces)
		}
		fmt.Printf("%s\t%12s\t%12s\t%12s\n", name, median, percentile5, percentile99)
	}
}

// TestBenchCGO benchmarks a minimal CGO call.
//
// Results on cpu:
//
//	go test . -test.v -test.run=Bench -test.count=1
//	Benchmarks:           Median         5%-tile        99%-tile
//	CGOCall                 38ns            38ns            40ns
func TestBenchCGO(t *testing.T) {
	plugin := must1(GetPlugin(*flagPluginName))
	testNames := []string{"CGOCall"}
	const repeats = 100
	testFns := []func(){
		func() {
			for _ = range repeats {
				dummyCGO(unsafe.Pointer(plugin.api))
			}
		},
	}
	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
}

// Benchmark tests arena
//
// Runtime in CPU:
//
//	Benchmarks:                           Median         5%-tile        99%-tile
//	TestBenchArena/arena/1                 153ns           144ns           170ns
//	TestBenchArena/arena/5                 156ns           154ns           165ns
//	TestBenchArena/arena/10                184ns           174ns           211ns
//	TestBenchArena/arena/100               547ns           535ns           671ns
//	TestBenchArena/arenaPool/1              86ns            83ns            92ns
//	TestBenchArena/arenaPool/5             101ns            99ns           109ns
//	TestBenchArena/arenaPool/10            121ns           119ns           135ns
//	TestBenchArena/arenaPool/100           487ns           477ns           528ns
//	TestBenchArena/malloc/1                135ns           132ns           143ns
//	TestBenchArena/malloc/5                512ns           508ns           541ns
//	TestBenchArena/malloc/10               984ns           975ns         1.104µs
//	TestBenchArena/malloc/100            9.401µs         9.346µs          9.82µs
//	TestBenchArena/go+pinner/1              88ns            85ns           138ns
//	TestBenchArena/go+pinner/5             317ns           308ns           582ns
//	TestBenchArena/go+pinner/10            646ns           594ns         2.291µs
//	TestBenchArena/go+pinner/100        10.437µs         6.754µs        38.438µs
func TestBenchArena(t *testing.T) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	numAllocationsList := []int{1, 5, 10, 100}
	allocations := make([]*int, 100)
	testNames := make([]string, 0, 4*len(numAllocationsList))
	testFns := make([]func(), 0, 4*len(numAllocationsList))
	const repeats = 10
	for _, allocType := range []string{"arena", "arenaPool", "malloc", "go+pinner"} {
		for _, numAllocations := range numAllocationsList {
			name := fmt.Sprintf("%s/%s/%d", t.Name(), allocType, numAllocations)
			testNames = append(testNames, name)
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
						arena := getArenaFromPool()
						for idx := range numAllocations {
							allocations[idx] = arenaAlloc[int](arena)
						}
						dummyCGO(unsafe.Pointer(allocations[numAllocations-1]))
						returnArenaToPool(arena)
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
			testFns = append(testFns, fn)
		}
	}
	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
}

// TestBenchBufferFromHost benchmarks host->buffer transfer time.
//
//	Benchmarks:                                                   Median         5%-tile        99%-tile
//	TestBenchBufferFromHost/shape=(Float32)[1 1]                   999ns           835ns         2.297µs
//	TestBenchBufferFromHost/shape=(Float32)[10 10]               1.115µs           852ns         2.813µs
//	TestBenchBufferFromHost/shape=(Float32)[100 100]             1.915µs          1.54µs         3.957µs
//	TestBenchBufferFromHost/shape=(Float32)[1000 1000]         129.644µs       128.748µs       144.503µs
func TestBenchBufferFromHost(t *testing.T) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	const repeats = 10
	numShapes := len(testShapes)
	inputData := make([][]float32, numShapes)
	testFns := make([]func(), numShapes)
	testNames := make([]string, numShapes)
	for shapeIdx, s := range testShapes {
		inputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			inputData[shapeIdx][i] = float32(i)
		}
		testNames[shapeIdx] = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx] = func() {
			for _ = range repeats {
				x := inputData[shapeIdx]
				s := testShapes[shapeIdx]
				buf := must1(ArrayToBuffer(client, x, s.Dimensions...))
				must(buf.Destroy())
			}
		}
	}
	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
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
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	const repeats = 10
	numShapes := len(testShapes)
	testFns := make([]func(), numShapes)
	testNames := make([]string, numShapes)
	// Prepare output data (host destination array) and upload buffers to GPU
	outputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	for shapeIdx, s := range testShapes {
		outputData[shapeIdx] = make([]float32, s.Size())
		for i := 0; i < s.Size(); i++ {
			outputData[shapeIdx][i] = float32(i)
		}
		buffers[shapeIdx] = must1(ArrayToBuffer(client, outputData[shapeIdx], s.Dimensions...))
		testNames[shapeIdx] = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx] = func() {
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

	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
}

// BenchmarkAdd1Execution benchmarks the execution time for a minimal program.
//
//	Benchmarks:                                                   Median         5%-tile        99%-tile
//	TestBenchAdd1Execution/shape=(Float32)[1 1]                  1.548µs          1.39µs         3.701µs
//	TestBenchAdd1Execution/shape=(Float32)[10 10]                1.468µs         1.297µs          3.11µs
//	TestBenchAdd1Execution/shape=(Float32)[100 100]              2.968µs         2.614µs         5.464µs
//	TestBenchAdd1Execution/shape=(Float32)[1000 1000]           38.035µs        36.275µs        79.621µs
func TestBenchAdd1Execution(t *testing.T) {
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	const repeats = 10
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	testNames := make([]string, numShapes)
	testFns := make([]func(), numShapes)
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
		testNames[shapeIdx] = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx] = func() {
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

	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
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
	plugin := must1(GetPlugin(*flagPluginName))
	client := must1(plugin.NewClient(nil))
	defer runtime.KeepAlive(client)

	// Prepare input data, the uploaded buffers and the executables.
	const repeats = 10
	numShapes := len(testShapes)
	execs := make([]*LoadedExecutable, numShapes)
	inputData := make([][]float32, numShapes)
	buffers := make([]*Buffer, numShapes)
	testNames := make([]string, numShapes)
	testFns := make([]func(), numShapes)
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
		testNames[shapeIdx] = fmt.Sprintf("%s/shape=%s", t.Name(), s)
		testFns[shapeIdx] = func() {
			for _ = range repeats {
				buf := buffers[shapeIdx]
				exec := execs[shapeIdx]
				output := must1(exec.Execute(buf).Done())[0]
				output.Destroy()
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

	benchmarkNamedFunctions(testNames, testFns, 10, repeats)
}
