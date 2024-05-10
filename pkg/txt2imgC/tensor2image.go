package sd

func sdTensorToImage(input *GGMLTensor) []byte {
	width := input.ne[0]
	height := input.ne[1]
	channels := input.ne[2]
	if channels != 3 || input.typ != GGMLTypeF32 {
		panic("invalid input tensor")
	}
	imageData := make([]byte, width*height*channels)
	for iy := 0; iy < int(height); iy++ {
		for ix := 0; ix < int(width); ix++ {
			for k := 0; k < int(channels); k++ {
				value := ggmlTensorGetF32(input, ix, iy, k)
				imageData[iy*int(width)*int(channels)+ix*int(channels)+k] = uint8(value * 255.0)
			}
		}
	}
	return imageData
}

type GGMLTensor struct {
	ne  [4]int64
	typ GGMLType
	// other fields omitted for brevity
}

type GGMLType int

const (
	GGMLTypeF32 GGMLType = iota
	// other types omitted for brevity
)

func ggmlTensorGetF32(tensor *GGMLTensor, x, y, z int) float32 {
	// implementation omitted for brevity
	return 0
}
