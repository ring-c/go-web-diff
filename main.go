package main

import (
	"os"

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

	// err = upscale(model)
	if err != nil {
		println(err.Error())
		return
	}
}

func generate(model *sd.Model) (err error) {
	err = model.LoadFromFile("/media/ed/files/sd/models/Stable-diffusion/ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
	if err != nil {
		println(err.Error())
		return
	}

	var params = sd.DefaultFullParams
	params.Width = 544
	params.Height = 960
	params.ClipSkip = 2
	params.CfgScale = 7
	params.SampleSteps = 24
	params.SampleMethod = opts.EULER_A
	params.Seed = 42
	params.BatchCount = 9

	err = model.Predict("score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, wolf, forest, full body", params)
	if err != nil {
		return
	}

	return
}

func upscale(model *sd.Model) (err error) {
	fileRead, err := os.Open("./output/0.png")
	if err != nil {
		return
	}

	defer func() {
		_ = fileRead.Close()
	}()

	fileWrite, err := os.Create("./output/1.png")
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

	return
}
