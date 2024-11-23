//go:build darwin

package xlabuilder_test

import (
	// Link CPU PJRT statically: required in Mac builds, since dynamically linking is not working.
	_ "github.com/gomlx/gopjrt/pjrt/static"
)
