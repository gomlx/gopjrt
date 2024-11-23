//go:build pjrt_cpu_static

package gopjrt

import (
	// Link CPU PJRT statically: slower but works on Mac.
	_ "github.com/gomlx/gopjrt/pjrt/cpu/static"
)
