package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"fmt"
	"k8s.io/klog/v2"
	"runtime"
	"unsafe"
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

func pjrtClientPlatformName(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformName_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_PlatformName(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_name, args.platform_name_size), nil
}

func pjrtClientPlatformVersion(plugin *Plugin, client *Client) (string, error) {
	args := C.new_PJRT_Client_PlatformVersion_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_PlatformVersion(plugin.api, args))
	if err != nil {
		return "", err
	}
	return cCharArray(args.platform_version, args.platform_version_size), nil
}

func pjrtClientProcessIndex(plugin *Plugin, client *Client) (int, error) {
	args := C.new_PJRT_Client_ProcessIndex_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_ProcessIndex(plugin.api, args))
	if err != nil {
		return -1, err
	}
	return int(args.process_index), nil
}

func pjrtClientDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_Devices_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_Devices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.devices), int(args.num_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(plugin, client, d)
	}
	return devices, nil
}

func pjrtClientAddressableDevices(plugin *Plugin, client *Client) ([]*Device, error) {
	args := C.new_PJRT_Client_AddressableDevices_Args()
	defer cFree(args)
	args.client = client.client
	err := toError(plugin, C.call_PJRT_Client_AddressableDevices(plugin.api, args))
	if err != nil {
		return nil, err
	}
	cDevices := cDataToSlice[*C.PJRT_Device](unsafe.Pointer(args.addressable_devices), int(args.num_addressable_devices))
	devices := make([]*Device, len(cDevices))
	for ii, d := range cDevices {
		devices[ii] = newDevice(plugin, client, d)
	}
	return devices, nil
}

// Client manages the resources of one device: its buffers, compilation and execution of HLO code.
type Client struct {
	plugin                    *Plugin
	client                    *C.PJRT_Client
	platform, platformVersion string
	processIndex              int
}

// newClient is called by Plugin.NewClient to create a new PJRT_Client wrapper.
func newClient(plugin *Plugin, client *C.PJRT_Client) *Client {
	c := &Client{plugin: plugin, client: client}
	var err error
	c.platform, err = pjrtClientPlatformName(plugin, c)
	if err != nil {
		klog.Errorf("Failed to retrieve client platform name (plugin %s): %v", plugin, err)
	}
	c.platformVersion, err = pjrtClientPlatformVersion(plugin, c)
	if err != nil {
		klog.Errorf("Failed to retrieve client platform version (plugin %s): %v", plugin, err)
	}
	c.processIndex, err = pjrtClientProcessIndex(plugin, c)
	if err != nil {
		klog.Errorf("Failed to retrieve client process index (plugin %s): %v", plugin, err)
	}
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

// String implements fmt.Stringer.
func (c *Client) String() string {
	if c.client == nil {
		return "Invalid client"
	}

	pid := c.ProcessIndex()
	var pidStr string
	if pid == 0 {
		pidStr = "single-process"
	} else {
		pidStr = fmt.Sprintf("pid=%d", pid)
	}
	return fmt.Sprintf("Client[plugin=%q, platform=%q, %s]", c.plugin.Name(), c.Platform()+" - "+c.PlatformVersion(), pidStr)
}

// Platform returns the name of the client platform.
func (c *Client) Platform() string {
	return c.platform
}

// PlatformVersion returns the version of the client platform.
func (c *Client) PlatformVersion() string {
	return c.platformVersion
}

// ProcessIndex returns the process index of the client platform.
// Always 0 in single-process settings.
func (c *Client) ProcessIndex() int {
	return c.processIndex
}

// Devices returns a list of all devices visible to the runtime, including addressable
// // and non-addressable devices.
func (c *Client) Devices() ([]*Device, error) {
	return pjrtClientDevices(c.plugin, c)
}

// AddressableDevices returns a list of devices addressable to the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment (Client.ProcessIndex() == 0).
func (c *Client) AddressableDevices() ([]*Device, error) {
	return pjrtClientAddressableDevices(c.plugin, c)
}
