//go:build pjrt_cpu_dynamic

package gopjrt

import (
	// Link (preload) CPU PJRT dynamically (as opposed to use `dlopen`).
	_ "github.com/gomlx/gopjrt/pjrt/cpu/dynamic"
)
