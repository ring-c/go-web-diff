package main

import (
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"os"

	"github.com/davecgh/go-spew/spew"
	"gorgonia.org/tensor"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func main() {
	spew.Config.Indent = "\t"

	var imgW = 1024
	var imgH = 768

	a := tensor.New(tensor.WithBacking(tensor.Random(tensor.Float32, imgW*imgH*3)), tensor.WithShape(imgW, imgH, 3))

	fmt.Printf("a:\n%v\n", a)

	data, err := tensorToImage(a)
	if err != nil {
		panic(err.Error())
		return
	}

	err = save(data)
	if err != nil {
		panic(err.Error())
		return
	}
}

func save(image image.Image) (err error) {
	var file *os.File
	file, err = os.Create("./output/test.png")
	if err != nil {
		return
	}

	err = imageToWriter(image, opts.PNG, file)
	if err != nil {
		return
	}

	err = file.Close()
	if err != nil {
		return
	}

	return
}

func imageToWriter(image image.Image, imageType opts.OutputsImageType, writer io.Writer) (err error) {
	err = errors.New("unknown image type")

	switch imageType {
	case opts.PNG:
		var enc = png.Encoder{CompressionLevel: png.BestSpeed}
		err = enc.Encode(writer, image)
	case opts.JPEG:
		err = jpeg.Encode(writer, image, &jpeg.Options{Quality: 100})
	}

	if err != nil {
		return
	}

	return
}
