package main

import (
	"errors"
	"image"
	"image/color"

	"gorgonia.org/tensor"
)

func tensorToImage(input *tensor.Dense) (img *image.RGBA, err error) {
	var shape = input.Shape()

	width, err := shape.DimSize(0)
	if err != nil {
		return
	}
	height, err := shape.DimSize(1)
	if err != nil {
		return
	}
	channels, err := shape.DimSize(2)
	if err != nil {
		return
	}

	if channels != 3 || input.Dtype() != tensor.Float32 {
		err = errors.New("invalid input tensor")
		return
	}

	img = image.NewRGBA(image.Rect(0, 0, width, height))

	for iy := 0; iy < height; iy++ {
		for ix := 0; ix < width; ix++ {
			var R, G, B uint8

			R, err = tensorGetF32(input, ix, iy, 0)
			if err != nil {
				return
			}

			G, err = tensorGetF32(input, ix, iy, 1)
			if err != nil {
				return
			}

			B, err = tensorGetF32(input, ix, iy, 2)
			if err != nil {
				return
			}

			img.Set(ix, iy, color.RGBA{
				R: R,
				G: G,
				B: B,
				A: 255,
			})
		}
	}

	return
}

func tensorGetF32(tx *tensor.Dense, x, y, z int) (data uint8, err error) {
	value, err := tx.At(x, y, z)
	if err != nil {
		return
	}

	val, ok := value.(float32)
	if !ok {
		err = errors.New("tensor value not float32")
		return
	}

	data = uint8(val * 255.0)

	return
}
