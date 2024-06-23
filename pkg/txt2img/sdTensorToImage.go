package txt2img

import (
	"image"
	"image/color"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (gen *Generator) sdTensorToImage(input unsafe.Pointer, in *opts.Options) (img *image.RGBA) {
	var channels = 3
	var byteData = make([]byte, in.Width*in.Height*channels)
	for iy := 0; iy < in.Height; iy++ {
		for ix := 0; ix < in.Width; ix++ {
			for k := 0; k < channels; k++ {
				value := gen.GGML.TensorGetF32(input, ix, iy, k, 0)
				byteData[iy*in.Width*channels+ix*channels+k] = uint8(value * 255.0)
			}
		}
	}

	img = image.NewRGBA(image.Rect(0, 0, in.Width, in.Height))

	for y := 0; y < in.Height; y++ {
		for x := 0; x < in.Width; x++ {
			idx := (y*in.Width + x) * 3
			img.Set(x, y, color.RGBA{
				R: byteData[idx],
				G: byteData[idx+1],
				B: byteData[idx+2],
				A: 255,
			})
		}
	}

	return
}
