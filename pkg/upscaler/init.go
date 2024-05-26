package upscaler

import (
	"github.com/ring-c/go-web-diff/pkg/bind"
)

type Upscaler struct {
}

func New() (*Upscaler, error) {
	libSd, _, err := bind.OpenLibrary()
	if err != nil {
		return nil, err
	}

	var impl = Upscaler{}

	// purego.RegisterLibFunc(&impl.sdSetLogCallback, libSd, "sd_set_log_callback")

	return &impl, err
}
