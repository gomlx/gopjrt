package xlabuilder_test

// This file implements specialized tests for the simple ops, which are not automatically
// generated in gen_simple_ops_test.go

import (
	. "github.com/gomlx/gopjrt/xlabuilder"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestAnd(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	lhs := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{7, -3}, 2))).Test(t)
	rhs := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{2, -1}, 2))).Test(t)
	output := capture(And(lhs, rhs)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []int8{2, -3} // Bitwise of 2-complement of negative numbers is a bit odd.
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{len(want)}, dims)
}

func TestOr(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	lhs := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{7, -3}, 2))).Test(t)
	rhs := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{8 + 2, -1}, 2))).Test(t)
	output := capture(Or(lhs, rhs)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []int8{15, -1} // Bitwise of 2-complement of negative numbers is a bit odd.
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{len(want)}, dims)
}

func TestShiftLeft(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{2, -1}, 2))).Test(t)
	n := capture(Constant(builder, NewScalarLiteral(int8(2)))).Test(t)
	output := capture(ShiftLeft(x, n)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []int8{8, -4}
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{2}, dims)
}

func TestShiftRightArithmetic(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{8, -4, 2, -2}, 4))).Test(t)
	n := capture(Constant(builder, NewScalarLiteral(int8(2)))).Test(t)
	output := capture(ShiftRightArithmetic(x, n)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []int8{2, -1, 0, -1 /* Yes, right shifting a negative number stops at -1 */}
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{len(want)}, dims)
}

func TestShiftRightLogical(t *testing.T) {
	client := getPJRTClient(t)
	builder := New(t.Name())

	x := capture(Constant(builder, mustNewArrayLiteral(t,
		[]int8{8, -8, 2, -2}, 4))).Test(t)
	n := capture(Constant(builder, NewScalarLiteral(int8(2)))).Test(t)
	output := capture(ShiftRightLogical(x, n)).Test(t)
	exec := compile(t, client, capture(builder.Build(output)).Test(t))
	want := []int8{2, 62, 0, 63}
	got, dims := execArrayOutput[int8](t, client, exec)
	require.Equal(t, want, got)
	require.Equal(t, []int{len(want)}, dims)
}
