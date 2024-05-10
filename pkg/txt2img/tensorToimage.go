package main

import (
	"errors"

	"gorgonia.org/tensor"
)

func tensorToImage(input *tensor.Dense) (data []byte, err error) {
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

	if channels != 3 { // || input.typ != GGMLTypeF32
		err = errors.New("invalid input tensor")
		return
	}

	data = make([]byte, width*height*channels)

	for iy := 0; iy < int(height); iy++ {
		for ix := 0; ix < int(width); ix++ {
			for k := 0; k < int(channels); k++ {
				var value float32
				value, err = tensorGetF32(input, ix, iy, k)
				if err != nil {
					return
				}

				data[iy*int(width)*int(channels)+ix*int(channels)+k] = uint8(value * 255.0)
			}
		}
	}

	return
}

func tensorGetF32(tx *tensor.Dense, x, y, z int) (data float32, err error) {
	value, err := tx.At(x, y, z)
	if err != nil {
		return
	}

	data, ok := value.(float32)
	if !ok {
		err = errors.New("tensor value not float32")
		return
	}

	return
}
