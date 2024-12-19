package pjrt

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestArena(t *testing.T) {
	arena := newArena(1024)
	for _ = range 2 {
		require.Equal(t, 1024, arena.size)
		require.Equal(t, 0, arena.current)
		_ = arenaAlloc[int](arena)
		require.Equal(t, 8, arena.current)
		_ = arenaAlloc[int32](arena)
		require.Equal(t, 16, arena.current)

		_ = arenaAllocSlice[byte](arena, 9) // Aligning, it will occupy 16 bytes total.
		require.Equal(t, 32, arena.current)

		require.Panics(t, func() { _ = arenaAlloc[[512]int](arena) }, "Arena out of memory")
		require.Panics(t, func() { _ = arenaAllocSlice[float64](arena, 512) }, "Arena out of memory")
		arena.Reset()
	}
	arena.Free()
}
