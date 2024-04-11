//go:build linux

package deps

import (
	_ "embed"
)

//go:embed linux/libsd-abi.so
var libStableDiffusion []byte

var libName = "libStableDiffusion-*.so"

func getLib(_ bool) []byte {
	return libStableDiffusion
}
