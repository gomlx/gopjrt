package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_new_struct.h"
#include "gen_api_calls.h"

*/
import "C"
import (
	"fmt"
	"github.com/pkg/errors"
	"os"
	"sync"
	"unsafe"
)

// Plugin represents a loaded PJRT plugin that can be used to compile and execute StableHLO code.
//
// Loaded plugins are singletons per platform and cached (GetPlugin will return a pointer to the same plugin if
// called with the same platform or its aliases).
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
//
// Design document: https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit
type Plugin struct {
	name, path string
	api        *C.PJRT_Api
	dllHandle  dllHandleWrapper
	attributes NamedValuesMap

	// UseStableHLO configures the plugin clients to convert XlaBuilder programs from "HLO" to "StableHLO"
	// before compilation. The "StableHLO" (encoded as MLIR) is the more recent
	// "intermediary representation" program language.
	//
	// Setting to true incurs in a conversion step from "HLO" to "StableHLO" during compilation. But some
	// PJRT will only support "StableHLO" (namely Apple Metal PJRT).
	//
	// Default is true, but it can be changed by setting the environment variable "GOPJRT_NO_STABLE_HLO=1"
	//
	// Most people don't need to worry about this, it should be an implementation detail.
	UseStableHLO bool

	// arenaPool reuses C-allocated buffers to pass parameters to the PJRT api. Pool shared across all clients.
	arenaPool *sync.Pool
}

// pjrtPluginInitialize calls C.PJRT_Plugin_Initialize.
func pjrtPluginInitialize(plugin *Plugin) error {
	args := C.new_PJRT_Plugin_Initialize_Args()
	defer cFree(args)
	args.extension_start = nil
	return toError(plugin, C.call_PJRT_Plugin_Initialize(plugin.api, args))
}

// pjrtPluginAttributes calls C.PJRT_Plugin_Attributes and returns the plugin's attributes.
func pjrtPluginAttributes(plugin *Plugin) (NamedValuesMap, error) {
	args := C.new_PJRT_Plugin_Attributes_Args()
	defer cFree(args)
	args.extension_start = nil
	err := toError(plugin, C.call_PJRT_Plugin_Attributes(plugin.api, args))
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to retrieve plugin attributes")
	}
	namedValues := cDataToSlice[C.PJRT_NamedValue](unsafe.Pointer(args.attributes), int(args.num_attributes))
	return pjrtNamedValuesToMap(namedValues), nil
}

// newPlugin creates a new plugin from the api pointer.
// Internal: use GetPlugin instead.
func newPlugin(name, pluginPath string, api *C.PJRT_Api, dllHandle dllHandleWrapper) (*Plugin, error) {
	plugin := &Plugin{
		name:         name,
		path:         pluginPath,
		api:          api,
		dllHandle:    dllHandle,
		UseStableHLO: os.Getenv("GOPJRT_NO_STABLE_HLO") == "",
		arenaPool: &sync.Pool{
			New: func() interface{} { return newArena(arenaDefaultSize) },
		},
	}
	err := pjrtPluginInitialize(plugin)
	if err != nil {
		return nil, errors.WithMessagef(err, "initializing PJRT Plugin %q", name)
	}
	plugin.attributes, err = pjrtPluginAttributes(plugin)
	if err != nil {
		return nil, errors.WithMessagef(err, "initializing PJRT Plugin %q", name)
	}
	return plugin, nil
}

// RegisterPreloadedPlugin can be used to register a PJRT plugin that has been pre-linked (dynamically or statically)
// with the binary -- as opposed to the usual loadPlugin using `dlopen` after the program has started.
//
// It takes as input the name to be associated with the plugin and an unsafe pointer (uintptr) to the API table
// returned by the plugin's C.GetPjrtApi().
//
// See sub-packages `cpu/static` and `cpu/dynamic` for examples of usage.
func RegisterPreloadedPlugin(name string, api uintptr) error {
	plugin, err := newPlugin(name, "_preloaded_", (*C.PJRT_Api)(unsafe.Pointer(api)), nil)
	if err != nil {
		return err
	}
	loadedPlugins[name] = plugin
	return nil
}

// GetPlugin returns the plugin with the given name -- typically it reflect the platform, e.g: "cpu" or "gpu".
// But one can also give the full path to the `.so` file with the plugin.
//
// Loaded plugins are singletons and cached (GetPlugin will return a pointer to the same plugin if
// called with the same name or its aliases).
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
// If it is not set it will search in `/usr/local/lib/gomlx` and the standard libraries directories of the
// system (in linux in LD_LIBRARY_CONFIG and /etc/ld.so.conf file).
func GetPlugin(name string) (*Plugin, error) {
	return loadNamedPlugin(name)
}

// Name returns the name of the plugin, usually it reflects its platform (cpu, gpu, tpu, etc.).
func (p *Plugin) Name() string {
	return p.name
}

// Path returns the path from where the plugin was loaded.
func (p *Plugin) Path() string {
	return p.path
}

// Version returns the version reported by the loaded plugin.
func (p *Plugin) Version() (major, minor int) {
	return int(p.api.pjrt_api_version.major_version), int(p.api.pjrt_api_version.minor_version)
}

// Attributes returns a NamedValueMap with the attributes returned by the plugin at the time of its initialization.
func (p *Plugin) Attributes() NamedValuesMap {
	return p.attributes
}

// String implements fmt.Stringer. It returns the platform and version of the plugin.
func (p *Plugin) String() string {
	major, minor := p.Version()
	if p.path == p.name {
		return fmt.Sprintf("PJRT plugin (%s) v%d.%d", p.Path(), major, minor)
	}
	return fmt.Sprintf("PJRT %q plugin (%s) v%d.%d", p.Name(), p.Path(), major, minor)
}

// NewClient creates a new Client object to manage available devices.
// The options (it can be left nil) are plugin specific, and should (but often aren't) documented by the plugins.
func (p *Plugin) NewClient(options NamedValuesMap) (*Client, error) {
	return newClient(p, options)
}

// getArenaFromPool gets an arena of the default size.
// Must be matched with a call returnArenaToPool when it's no longer used.
func (p *Plugin) getArenaFromPool() *arenaContainer {
	return p.arenaPool.Get().(*arenaContainer)
}

// returnArenaToPool returns an arena acquired with getArenaFromPool.
// It also resets the arena.
func (p *Plugin) returnArenaToPool(a *arenaContainer) {
	a.Reset()
	p.arenaPool.Put(a)
}
