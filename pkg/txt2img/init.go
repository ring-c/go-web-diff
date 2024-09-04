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

	Text2Image func(
		sdCTX unsafe.Pointer,
		prompt, negPrompt string,
		clipSkip int,
		cfgScale, guidance float64,
		width, height int,
		sampleMethod int,
		sampleSteps int,
		seed int64,
		batchCount int,
		controlCond unsafe.Pointer,
		controlStrength, styleStrength float64,
		normalizeInput bool,
		inputIDImagesPath string,
	) unsafe.Pointer

	// GoSample func(sdCTX, ggmlCTX, xT unsafe.Pointer, prompt string, sigmasCnt int, sigmas []float32) unsafe.Pointer // ggml_tensor
	// DecodeFirstStage func(sdCTX, ggmlCTX, inputTX, outputTX unsafe.Pointer)
	// ApplyLora func(sdCTX unsafe.Pointer, lora string)
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
	purego.RegisterLibFunc(&impl.Text2Image, libSd, "txt2img")

	return &impl, err
}
