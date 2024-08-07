package txt2img

import (
	"sync"
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

	fileWrite sync.WaitGroup
	filenames []string

	// SetLearnedCondition func(sdCTX, ggmlCTX unsafe.Pointer, prompt string, width, height, clipSkip int)         // pair
	GoSample func(sdCTX, ggmlCTX, xT unsafe.Pointer, prompt string, sigmasCnt int, sigmas []float32) unsafe.Pointer // ggml_tensor

	DecodeFirstStage func(sdCTX, ggmlCTX, inputTX, outputTX unsafe.Pointer)

	ApplyLora func(sdCTX unsafe.Pointer, lora string)
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

	purego.RegisterLibFunc(&impl.GGML.Init, libSd, "go_ggml_init")
	purego.RegisterLibFunc(&impl.GGML.Free, libSd, "ggml_free")
	// purego.RegisterLibFunc(&impl.GGML.TensorOverhead, libSd, "ggml_tensor_overhead")
	purego.RegisterLibFunc(&impl.GGML.NewTensor4D, libSd, "ggml_new_tensor_4d")
	purego.RegisterLibFunc(&impl.GGML.TensorSetF32, libSd, "go_ggml_tensor_set_f32")
	purego.RegisterLibFunc(&impl.GGML.TensorSetF32Rand, libSd, "go_ggml_tensor_set_f32_randn")

	purego.RegisterLibFunc(&impl.GGML.TensorScale, libSd, "go_ggml_tensor_scale")
	purego.RegisterLibFunc(&impl.GGML.TensorScaleOutput, libSd, "go_ggml_tensor_scale_output")
	purego.RegisterLibFunc(&impl.GGML.TensorClamp, libSd, "go_ggml_tensor_clamp")

	purego.RegisterLibFunc(&impl.GGML.TensorGetF32, libSd, "go_ggml_tensor_get_f32")
	// purego.RegisterLibFunc(&impl.GGML.VectorToGgmlTensorI32, libSd, "go_vector_to_ggml_tensor_i32")

	// purego.RegisterLibFunc(&impl.SetLearnedCondition, libSd, "go_set_learned_condition")
	purego.RegisterLibFunc(&impl.ApplyLora, libSd, "apply_lora")

	purego.RegisterLibFunc(&impl.GoSample, libSd, "go_sample")
	purego.RegisterLibFunc(&impl.DecodeFirstStage, libSd, "go_decode_first_stage")

	return &impl, err
}
