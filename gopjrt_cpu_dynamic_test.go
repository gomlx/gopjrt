//go:build pjrt_cpu_dynamic

package gopjrt_test

import (
	// Link (preload) CPU PJRT dynamically (as opposed to use `dlopen`).
	_ "github.com/gomlx/gopjrt/pjrt/cpu/dynamic"
)
