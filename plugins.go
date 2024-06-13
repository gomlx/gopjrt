package gopjrt

/*
#include "pjrt_c_api.h"
#include "gen_new_struct.h"
#include "gen_api_calls.h"

*/
import "C"
import (
	"fmt"
	"github.com/pkg/errors"
)

var ()

// Plugin represents a loaded PJRT plugin that can be used to compile and execute StableHLO code.
//
// Loaded plugins are singletons per platform and cached (GetPlugin will return a pointer to the same plugin if
// called with the same platform or its aliases).
type Plugin struct {
	platform string
	api      *C.PJRT_Api
}

// newPlugin creates a new plugin from the api pointer.
// Internal: use GetPlugin instead.
func newPlugin(platform string, api *C.PJRT_Api) (*Plugin, error) {
	plugin := &Plugin{platform: platform, api: api}

	// Make sure the initialization call succeed.
	initArgs := C.new_PJRT_Plugin_Initialize_Args()
	initArgs.extension_start = nil
	pjrtErr := C.call_PJRT_Plugin_Initialize(plugin.api, initArgs)
	if pjrtErr != nil {
		return nil, errors.Errorf("Failed to initialize plugin") // include error message.
	}
	return plugin, nil
}

// GetPlugin returns the plugin for the given platform, or if one is not loaded yet, attempts to load it.
//
// Loaded plugins are singletons per platform and cached (GetPlugin will return a pointer to the same plugin if
// called with the same platform or its aliases).
func GetPlugin(platform string) (*Plugin, error) {
	return loadPlatformPlugin(platform)
}

// Platform returns the platform of the plugin.
func (p *Plugin) Platform() string {
	return p.platform
}

// Version returns the version reported by the loaded plugin.
func (p *Plugin) Version() (major, minor int) {
	return int(p.api.pjrt_api_version.major_version), int(p.api.pjrt_api_version.minor_version)
}

// String implements fmt.Stringer. It returns the platform and version of the plugin.
func (p *Plugin) String() string {
	major, minor := p.Version()
	return fmt.Sprintf("PJRT %s Plugin v%d.%d", p.Platform(), major, minor)
}
