package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"
import (
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
)

// Event is a reference that a future event (when something is done), and it is created by asynchronous calls.
//
// While it is exported here if someone needs it to implement some extension, usually users of the Go's pjrt package
// don't need to use it directly: the various methods of the API handles the events.
type Event struct {
	wrapper *eventWrapper
}

type eventWrapper struct {
	c      *C.PJRT_Event
	plugin *Plugin
}

// newEvent creates Event and registers it for freeing.
func newEvent(plugin *Plugin, cEvent *C.PJRT_Event) *Event {
	e := &Event{&eventWrapper{
		plugin: plugin,
		c:      cEvent,
	}}

	runtime.AddCleanup(e, func(wrapper *eventWrapper) {
		err := e.Destroy()
		if err != nil {
			klog.Errorf("pjrt.Event.Destroy failed: %+v", err)
		}
	}, e.wrapper)
	return e
}

func (wrapper *eventWrapper) Destroy() error {
	if wrapper == nil || wrapper.plugin == nil || wrapper.c == nil {
		// Already destroyed, no-op.
		return nil
	}
	defer runtime.KeepAlive(wrapper)
	args := C.new_PJRT_Event_Destroy_Args()
	defer cFree(args)
	args.event = wrapper.c
	err := toError(wrapper.plugin, C.call_PJRT_Event_Destroy(wrapper.plugin.api, args))
	wrapper.plugin = nil
	wrapper.c = nil
	return err
}

func (wrapper *eventWrapper) IsValid() bool {
	return wrapper != nil && wrapper.c != nil && wrapper.plugin != nil
}

// Destroy the Event and release its resources.
// After that Event is no longer valid.
// This is automatically called if Event is garbage collected.
func (e *Event) Destroy() error {
	return e.wrapper.Destroy()
}

// Await blocks the calling thread until `event` is ready, then returns the error, if any.
func (e *Event) Await() error {
	if e == nil || !e.wrapper.IsValid() {
		return errors.New("Event is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	defer runtime.KeepAlive(e)
	args := C.new_PJRT_Event_Await_Args()
	defer cFree(args)
	args.event = e.wrapper.c
	return toError(e.wrapper.plugin, C.call_PJRT_Event_Await(e.wrapper.plugin.api, args))
}

// AwaitAndFree blocks the calling thread until `event` is ready, destroy the even and then returns the error, if any.
//
// An error destroying the even is simply reported in the logs, but not returned.
func (e *Event) AwaitAndFree() error {
	if e == nil || !e.wrapper.IsValid() {
		return errors.New("Event is nil, or its plugin or wrapped C representation is nil -- has it been destroyed already?")
	}
	err := e.Await()
	err2 := e.Destroy()
	if err2 != nil {
		klog.Errorf("Error destroying an event already waited: %+v", err2)
	}
	return err
}
