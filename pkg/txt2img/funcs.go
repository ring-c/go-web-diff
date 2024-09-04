package txt2img

import (
	"image"
	"image/color"
	"unsafe"
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

func goString(c uintptr) string {
	// We take the address and then dereference it to trick go vet from creating a possible misuse of unsafe.Pointer
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&c))
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Add(ptr, uintptr(length))) == '\x00' {
			break
		}
		length++
	}
	return unsafe.String((*byte)(ptr), length)
}

func goImageSlice(c unsafe.Pointer, size int) []Image {
	// We take the address and then dereference it to trick go vet from creating a possible misuse of unsafe.Pointer
	ptr := *&c
	if ptr == nil {
		return nil
	}
	img := (*cImage)(ptr)
	goImages := make([]Image, 0, size)
	imgSlice := unsafe.Slice(img, size)
	for _, image := range imgSlice {
		var gImg Image
		gImg.Channel = image.channel
		gImg.Width = image.width
		gImg.Height = image.height
		dataPtr := *(*unsafe.Pointer)(unsafe.Pointer(&image.data))
		if ptr != nil {
			gImg.Data = unsafe.Slice((*byte)(dataPtr), image.channel*image.width*image.height)
		}
		goImages = append(goImages, gImg)
	}
	return goImages
}

type cImage struct {
	width   uint32
	height  uint32
	channel uint32
	data    uintptr
}

type Image struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    []byte
}
