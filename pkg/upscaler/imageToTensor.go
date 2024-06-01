package upscaler

import (
	"image"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/bind"
)

func (u *Upscaler) imageToTensor(img image.Image, input unsafe.Pointer) {
	width := img.Bounds().Max.X
	height := img.Bounds().Max.Y
	channels := 3
	imageData := bind.ImageToBytes(img)

	for iy := 0; iy < height; iy++ {
		for ix := 0; ix < width; ix++ {
			for k := 0; k < channels; k++ {
				value := float64(imageData.Data[iy*width*channels+ix*channels+k]) / 255.0
				u.GGML.TensorSetF32(input, value, ix, iy, k, 0)
			}
		}
	}
}
