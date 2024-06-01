package upscaler

import (
	"github.com/ring-c/go-web-diff/pkg/bind"
)

type Upscaler struct {
	// Upscale func(ctx unsafe.Pointer, upscaleFactor, width, height, channel uint32, data []byte) unsafe.Pointer
}

func New() (*Upscaler, error) {
	libSd, _, err := bind.OpenLibrary()
	if err != nil {
		return nil, err
	}

	var impl = Upscaler{}

	// purego.RegisterLibFunc(&impl.Upscale, libSd, "upscale_go")

	return &impl, err
}
