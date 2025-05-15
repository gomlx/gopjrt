// Package protos is empty, it simply include a rule to generate all the sub-packages:
// one sub-package per XLA proto used in gopjrt.
package protos

//go:generate go run ../cmd/protoc_xla_protos
