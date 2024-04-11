//go:build darwin || linux

package bind

import (
	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/bind/deps"
)

func openLibrary() (ptr uintptr, filename string, err error) {
	filename, err = deps.DumpSDLibrary(true)
	if err != nil {
		return
	}

	ptr, err = purego.Dlopen(filename, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return
	}

	return
}

func closeLibrary(handle uintptr) error {
	return purego.Dlclose(handle)
}
