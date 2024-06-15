package pjrt

// This file holds the definition of functions commonly used in different parts.

// keys returns the keys of a map in the form of a slice.
func keys[K comparable, V any](m map[K]V) []K {
	s := make([]K, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	return s
}
