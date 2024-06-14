package gopjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"k8s.io/klog/v2"
	"runtime"
)

func pjrtClientCreate(plugin *Plugin, options NamedValuesMap) (*Client, error) {
	args := C.new_PJRT_Client_Create_Args()
	defer cFree(args)
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
	c := &Client{plugin: plugin, client: client}
	runtime.SetFinalizer(c, func(c *Client) {
		err := c.Destroy()
		if err != nil {
			klog.Errorf("Client.Destroy failed: %v", err)
		}
	})
	return c
}

// Destroy the client, release resources, and Client is no longer valid.
// This is automatically called if Client is garbage collected.
func (c *Client) Destroy() error {
	if c.plugin == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(c)
	args := C.new_PJRT_Client_Destroy_Args()
	defer cFree(args)
	args.client = c.client
	err := toError(c.plugin, C.call_PJRT_Client_Destroy(c.plugin.api, args))
	c.plugin = nil
	return err
}
