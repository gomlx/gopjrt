package pjrt

/*
#include "pjrt_c_api.h"
#include "gen_api_calls.h"
#include "gen_new_struct.h"
*/
import "C"

// LoadedExecutable is a reference to a compiled program ready to be executed.
type LoadedExecutable struct {
	cLoadedExecutable *C.PJRT_LoadedExecutable
	plugin            *Plugin
	client            *Client
}
