package txt2img

import (
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/bind"
	"github.com/ring-c/go-web-diff/pkg/ggml"
	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"

	_ "github.com/ianlancetaylor/cgosymbolizer"
)

type Generator struct {
	GGML  ggml.Struct
	Model *sd.Model

	// GetLearnedCondition func(sdCTX, ggmlCTX unsafe.Pointer, prompt string, width, height, clipSkip int) unsafe.Pointer // pair
	// PairGet  func(pair unsafe.Pointer, first bool) unsafe.Pointer                                                // ggml_tensor
	GoSample func(sdCTX, ggmlCTX, xT unsafe.Pointer, sigmasCnt int, sigmas []float32) unsafe.Pointer // ggml_tensor
}

func New(in *opts.Options) (*Generator, error) {
	libSd, _, err := bind.OpenLibrary()
	if err != nil {
		return nil, err
	}

	model, err := sd.NewModel(in)
	if err != nil {
		return nil, err
	}

	err = model.LoadFromFile(in.ModelPath)
	if err != nil {
		return nil, err
	}

	var impl = Generator{
		GGML:  ggml.Struct{},
		Model: model,
	}

	purego.RegisterLibFunc(&impl.GGML.InitGo, libSd, "ggml_init_go")
	// purego.RegisterLibFunc(&impl.GGML.TensorOverhead, libSd, "ggml_tensor_overhead")
	purego.RegisterLibFunc(&impl.GGML.NewTensor4D, libSd, "ggml_new_tensor_4d")
	purego.RegisterLibFunc(&impl.GGML.TensorSetF32, libSd, "go_ggml_tensor_set_f32")
	purego.RegisterLibFunc(&impl.GGML.TensorSetF32Rand, libSd, "go_ggml_tensor_set_f32_randn")

	// purego.RegisterLibFunc(&impl.GetLearnedCondition, libSd, "go_get_learned_condition")
	// purego.RegisterLibFunc(&impl.PairGet, libSd, "go_pair_get")
	purego.RegisterLibFunc(&impl.GoSample, libSd, "go_sample")

	return &impl, err
}
