package xlabuilder_test

import (
	"fmt"
	"github.com/gomlx/gopjrt/dtypes"
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestTuple(t *testing.T) {
	// f(x) = [x^2, sqrt(x)]
	builder := New("x*x, sqrt(x)")
	x, err := Parameter(builder, "x", 0, MakeShape(dtypes.F32)) // Scalar float32.
	require.NoError(t, err)
	x2, err := Mul(x, x)
	require.NoError(t, err)
	sqrtX, err := Sqrt(x)
	require.NoError(t, err)
	fX, err := Tuple(x2, sqrtX)
	require.NoError(t, err)

	// Get computation created.
	comp, err := builder.Build(fX)
	require.NoError(t, err)
	fmt.Printf("HloModule proto:\n%s\n\n", comp.TextHLO())

	stableHLO := comp.SerializedHLO()
	defer stableHLO.Free()
	if *flagStableHLOOutput != "" {
		f, err := os.Create(*flagStableHLOOutput)
		require.NoErrorf(t, err, "Failed to open StableHLO proto output file %q", *flagStableHLOOutput)
		bufBytes := stableHLO.Bytes()
		n, err := f.Write(bufBytes)
		require.NoErrorf(t, err, "Failed to write StableHLO proto output file %q", *flagStableHLOOutput)
		require.Equal(t, len(bufBytes), n)
		require.NoError(t, f.Close(), "Failed to close StableHLO proto output file %q", *flagStableHLOOutput)
	}
}

func TestConstants(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// f(x)=x+1
	x := capture(Parameter(builder, "x", 0, MakeShape(dtypes.F32))).Test(t) // Scalar float32.
	one := capture(Constant(builder, NewScalarLiteral(float32(1)))).Test(t)
	fX := capture(Add(x, one)).Test(t)
	comp := capture(builder.Build(fX)).Test(t)

	// Check values.
	addOne := compile(t, client, comp)
	require.InDelta(t, float32(2), execWithScalars(t, client, addOne, float32(1)), 1e-3)
	require.InDelta(t, float32(8), execWithScalars(t, client, addOne, float32(7)), 1e-3)

	// f(x)=x+1 with broadcast
	x = capture(Parameter(builder, "x", 0, MakeShape(dtypes.Int64, 3))).Test(t) // Scalar float32.
	one = capture(Constant(builder, NewScalarLiteral(int64(1)))).Test(t)
	fX = capture(Add(x, one)).Test(t)
	comp = capture(builder.Build(fX)).Test(t)

	// Check values.
	addOne = compile(t, client, comp)
	got, dims := execWithSlices(t, client, addOne, []int64{1, 7, 13})
	require.Equal(t, []int64{1 + 1, 7 + 1, 13 + 1}, got)
	require.Equal(t, []int{3}, dims)
}

func TestIota(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	exec := compile(t, client, capture(builder.Build(iotaOp)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)

	iotaOp = capture(Iota(builder, MakeShape(dtypes.F64, 2, 3), 1)).Test(t)
	exec = compile(t, client, capture(builder.Build(iotaOp)).Test(t))
	got, dims = execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 1, 2, 0, 1, 2}, got)
	require.Equal(t, []int{2, 3}, dims)

}

func TestIdentity(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, just with an identity op in between.
	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	identityOp := Identity(iotaOp)
	exec := compile(t, client, capture(builder.Build(identityOp)).Test(t))
	got, dims := execArrayOutput[float64](t, client, exec)
	require.Equal(t, []float64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)
}

func TestConvertDType(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, but change the dtype of the result.
	iotaOp := capture(Iota(builder, MakeShape(dtypes.F64, 3, 2), 0)).Test(t)
	output := capture(ConvertDType(iotaOp, dtypes.Int64)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[int64](t, client, exec)
	require.Equal(t, []int64{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{3, 2}, dims)
}

func TestWhere(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	// Exact same test as iota, but change the dtype of the result.
	shape := MakeShape(dtypes.Float32, 5, 3)
	zeros := capture(Constant(builder, NewLiteralFromShape(shape))).Test(t)
	one := capture(Constant(builder, NewScalarLiteral(float32(1)))).Test(t)
	ones := capture(Add(zeros, one)).Test(t)
	values := capture(Iota(builder, shape, 0)).Test(t)
	two := capture(Constant(builder, NewScalarLiteral(float32(2)))).Test(t)
	greaterThanTwo := capture(GreaterThan(values, two)).Test(t)
	output := capture(Where(greaterThanTwo, ones, zeros)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, []float32{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		1, 1, 1,
		1, 1, 1,
	}, got)
	require.Equal(t, shape.Dimensions, dims)
}

func TestReshape(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Int8, 3, 2), 0)).Test(t)
	_, err := Reshape(input, 7) // Bad reshape.
	require.Error(t, err)
	output := capture(Reshape(input, 6, 1, 1)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, []int8{0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{6, 1, 1}, dims)
}

func TestBroadcast(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	input := capture(Iota(builder, MakeShape(dtypes.Float32, 3, 2), 0)).Test(t)
	output := capture(Broadcast(input, 2)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	got, dims := execArrayOutput[float32](t, client, exec)
	require.Equal(t, []float32{0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2}, got)
	require.Equal(t, []int{2, 3, 2}, dims)
}
