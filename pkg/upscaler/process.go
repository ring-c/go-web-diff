package upscaler

import (
	"image"
	"os"
	"path/filepath"

	"github.com/davecgh/go-spew/spew"

	"github.com/ring-c/go-web-diff/pkg/generate"
)

func (u *Upscaler) Process(in *generate.InputData, file string) (err error) {

	var filenameIn = filepath.Join(in.Params.OutputDir, file)
	// var filenameOut = filenameIn
	// if !in.Params.DeleteUpscaled {
	// 	filenameOut = filepath.Join(in.Params.OutputDir, "u-"+file)
	// }

	var fileRead *os.File
	fileRead, err = os.Open(filenameIn)
	if err != nil {
		return
	}

	defer func() {
		_ = fileRead.Close()
	}()

	decoded, _, err := image.Decode(fileRead)
	if err != nil {
		return
	}

	// upscale

	result, err := u.upscale(decoded)
	if err != nil {
		return
	}

	println("\n\n\nRESULT\n\n")

	spew.Dump(result)

	// fileWrite, err := os.Create(filenameOut)
	// if err != nil {
	// 	return
	// }
	//
	// defer func() {
	// 	_ = fileWrite.Close()
	// }()
	//
	// err = imageToWriter(img, fileWrite)
	// if err != nil {
	// 	return
	// }

	return
}
