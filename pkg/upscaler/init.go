package upscaler

import (
	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/bind"
)

func New() (*Upscaler, error) {
	libSd, _, err := bind.OpenLibrary()
	if err != nil {
		return nil, err
	}

	var impl = Upscaler{}

	purego.RegisterLibFunc(&impl.GGML.Init, libSd, "ggml_init")

	return &impl, err
}
