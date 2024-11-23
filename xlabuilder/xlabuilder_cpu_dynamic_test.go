//go:build pjrt_cpu_dynamic

package xlabuilder_test

import (
	// Link CPU PJRT statically: slower but works on Mac.
	_ "github.com/gomlx/gopjrt/pjrt/cpu/dynamic"
)
