package bind

import (
	"image"
	"image/color"
)

func bytesToImage(byteData []byte, width, height int) (img *image.RGBA) {
	img = image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := (y*width + x) * 3
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

type Image struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    []byte
}

func ImageToBytes(decode image.Image) Image {
	bounds := decode.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y
	size := width * height * 3
	bytesImg := make([]byte, size)

	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			idx := (y*width + x) * 3
			r, g, b, _ := decode.At(x, y).RGBA()
			bytesImg[idx] = byte(r >> 8)
			bytesImg[idx+1] = byte(g >> 8)
			bytesImg[idx+2] = byte(b >> 8)
		}
	}

	return Image{
		Width:   uint32(width),
		Height:  uint32(height),
		Data:    bytesImg,
		Channel: 3,
	}
}
