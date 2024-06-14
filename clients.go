package gopjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"

func pjrtClientCreate(plugin *Plugin, options NamedValuesMap) (*Client, error) {
	args := C.new_PJRT_Client_Create_Args()
	defer cFree(args)
	args.extension_start = nil
	args.create_options, args.num_options = options.mallocArrayPJRT_NamedValue()
	// No callback support yet, so we leave the various PJRT_KeyValue... fields empty.

	err := toError(plugin, C.call_PJRT_Client_Create(plugin.api, args))
	if err != nil {
		return nil, err
	}
	return newClient(plugin, args.client), nil
}

// Client manages the resources of one device: its buffers, compilation and execution of HLO code.
type Client struct {
	plugin *Plugin
	client *C.PJRT_Client
}

// newClient is called by Plugin.NewClient to create a new PJRT_Client wrapper.
func newClient(plugin *Plugin, client *C.PJRT_Client) *Client {
	return &Client{plugin: plugin, client: client}
}
