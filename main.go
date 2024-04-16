package main

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/davecgh/go-spew/spew"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
)

func main() {
	spew.Config.Indent = "\t"

	options := sd.DefaultOptions
	options.GpuEnable = true
	options.Wtype = opts.F16
	// options.Schedule = opts.KARRAS
	options.Debug = true

	model, err := sd.NewModel(options)
	if err != nil {
		println(err.Error())
		return
	}
	defer func() {
		_ = model.Close()
	}()

	// println(model.GetSystemInfo())

	err = generate(model)
	if err != nil {
		println(err.Error())
		return
	}

	err = upscale(model)
	if err != nil {
		println(err.Error())
		return
	}
}

func generate(model *sd.Model) (err error) {
	params, err := sd.GetInput()
	if err != nil {
		return
	}

	err = model.LoadFromFile("/media/ed/files/sd/models/Stable-diffusion/ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
	if err != nil {
		return
	}

	err = model.Predict(params)
	if err != nil {
		return
	}

	return
}

func upscale(model *sd.Model) (err error) {
	var outputDir = "./output/"

	dir, err := os.ReadDir(outputDir)
	if err != nil {
		return
	}

	for _, file := range dir {
		if strings.HasPrefix(file.Name(), "u-") {
			continue
		}

		var filename = filepath.Join(outputDir, file.Name())
		var filenameOut = filepath.Join(outputDir, "u-"+file.Name())

		var fileRead *os.File
		fileRead, err = os.Open(filename)
		if err != nil {
			return
		}

		defer func() {
			_ = fileRead.Close()
		}()

		var fileWrite *os.File
		fileWrite, err = os.Create(filenameOut)
		if err != nil {
			return
		}

		defer func() {
			_ = fileWrite.Close()
		}()

		err = model.UpscaleImage(fileRead, "/media/ed/files/sd/models/ESRGAN/RealESRGAN_x4plus_anime_6B.pth", 2, fileWrite)
		if err != nil {
			return
		}

		err = os.Remove(filename)
		if err != nil {
			return
		}
	}

	return
}
