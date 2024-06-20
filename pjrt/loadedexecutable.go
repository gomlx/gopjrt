package pjrt

// LoadedExecutable is a reference to a compiled program ready to be executed.
type LoadedExecutable struct {
	plugin *Plugin
	client *Client
}
