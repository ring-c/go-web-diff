package txt2img

import (
	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/bind"
	"github.com/ring-c/go-web-diff/pkg/ggml"
)

type Generator struct {
	GGML ggml.Struct
}

func New() (*Generator, error) {
	libSd, _, err := bind.OpenLibrary()
	if err != nil {
		return nil, err
	}

	var impl = Generator{
		GGML: ggml.Struct{},
	}

	purego.RegisterLibFunc(&impl.GGML.InitGo, libSd, "ggml_init_go")
	// purego.RegisterLibFunc(&impl.GGML.TensorOverhead, libSd, "ggml_tensor_overhead")
	purego.RegisterLibFunc(&impl.GGML.NewTensor4D, libSd, "ggml_new_tensor_4d")
	purego.RegisterLibFunc(&impl.GGML.TensorSetF32, libSd, "ggml_tensor_set_f32_go")

	return &impl, err
}
