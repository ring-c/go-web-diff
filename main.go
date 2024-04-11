package main

import (
	"io"
	"os"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
)

func main() {

	options := sd.DefaultOptions
	options.GpuEnable = true
	options.Wtype = opts.F16
	options.Schedule = opts.KARRAS

	model, err := sd.NewModel(options)
	if err != nil {
		println(err.Error())
		return
	}
	defer func() {
		_ = model.Close()
	}()

	err = model.LoadFromFile("/media/ed/files/sd/models/Stable-diffusion/dreamshaperXL_v21TurboDPMSDE.safetensors")
	if err != nil {
		println(err.Error())
		return
	}

	filenames := []string{
		"./output/0.png",
	}

	var writers []io.Writer
	for _, filename := range filenames {
		var file *os.File
		file, err = os.Create(filename)
		if err != nil {
			println(err.Error())
			return
		}

		writers = append(writers, file)
		defer func() {
			_ = file.Close()
		}()
	}

	var params = sd.DefaultFullParams
	params.Width = 1024
	params.Height = 1024
	params.CfgScale = 2
	params.SampleSteps = 32
	params.SampleMethod = opts.EULER_A
	params.BatchCount = len(writers)

	err = model.Predict("1wolf, forest, full body", params, writers)
	if err != nil {
		println(err.Error())
		return
	}

	/*
		fileRead, err := os.Open("./output/1.png")
		if err != nil {
			println(err.Error())
			return
		}

		fileWrite, err := os.Create("./output/2.png")
		if err != nil {
			println(err.Error())
			return
		}

		err = model.UpscaleImage(fileRead, "/media/ed/files/sd/models/ESRGAN/RealESRGAN_x4plus_anime_6B.pth", 2, fileWrite)
		if err != nil {
			println(err.Error())
			return
		}
	*/
}
