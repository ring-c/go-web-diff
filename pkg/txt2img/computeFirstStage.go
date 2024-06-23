package txt2img

import (
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (gen *Generator) computeFirstStage(workCtx, input unsafe.Pointer, in *opts.Options) (result unsafe.Pointer) {

	// VERSION_XL
	var scaleFactor float32 = 0.13025
	gen.GGML.TensorScale(input, 1.0/scaleFactor)

	result = gen.GGML.NewTensor4D(workCtx, 0, in.Width, in.Height, 3, 1)

	gen.DecodeFirstStage(gen.Model.GetCTX(), workCtx, input, result)

	gen.GGML.TensorScaleOutput(result)
	gen.GGML.TensorClamp(result, 0.0, 1.0)

	return
}
