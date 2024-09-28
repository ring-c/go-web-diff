package sd

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path"
	"time"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func resultCallback(num uint64, imageData *byte, gen *opts.Generation) {
	go func(num uint64, imageData *byte, gen *opts.Generation) {
		defer gen.OneDone()

		var filename = fmt.Sprintf("%d-%d.png", time.Now().Unix(), gen.Seed+int64(num-1))
		file, err := os.Create(path.Join(gen.OutputDir, filename))
		if err != nil {
			println("resultCallback os.Create: " + err.Error())
			return
		}

		resultWrite(file, imageData, gen.Width, gen.Height)

		err = file.Close()
		if err != nil {
			println("resultCallback file.Close: " + err.Error())
			return
		}

		gen.AddFilename(filename)
	}(num, imageData, gen)
}

func resultStepCallback(num, step uint64, imageData *byte, gen *opts.Generation) {
	go func(step uint64, imageData *byte, gen *opts.Generation) {
		// defer gen.OneDone()

		var filename = fmt.Sprintf("%d-%d-step%d.png", time.Now().Unix(), gen.Seed+int64(num-1), step)
		file, err := os.Create(path.Join(gen.OutputDir, filename))
		if err != nil {
			println("resultStepCallback os.Create: " + err.Error())
			return
		}

		resultWrite(file, imageData, gen.Width, gen.Height)

		err = file.Close()
		if err != nil {
			println("resultStepCallback file.Close: " + err.Error())
			return
		}

		// gen.AddFilename(filename)
	}(step, imageData, gen)
}

func resultWrite(file *os.File, imageData *byte, width, height int) {
	var data = unsafe.Slice(imageData, width*height*3)

	var outputImg = bytesToImage(data, width, height)
	var err = imageToWriter(outputImg, file)
	if err != nil {
		println("resultWrite imageToWriter: " + err.Error())
		return
	}
}

func imageToWriter(image *image.RGBA, writer io.Writer) (err error) {
	var enc = png.Encoder{
		CompressionLevel: png.BestSpeed,
	}

	err = enc.Encode(writer, image)
	if err != nil {
		return
	}

	return
}

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
