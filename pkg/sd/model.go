package sd

import (
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"time"

	"github.com/ring-c/go-web-diff/pkg/bind"
	"github.com/ring-c/go-web-diff/pkg/opts"
)

type Model struct {
	options *opts.Options

	cSD *bind.CStableDiffusionImpl
	ctx *bind.CStableDiffusionCtx

	esrganPath  string
	upscalerCtx *bind.CUpScalerCtx

	// diffusionModelPath string
	// isAutoLoad         bool
	// dylibPath          string
}

func NewModel(options *opts.Options) (model *Model, err error) {
	csd, err := bind.NewCStableDiffusion()
	if err != nil {
		return
	}

	if options.Debug {
		csd.SetLogCallBack(func(level opts.LogLevel, text string) {
			fmt.Printf("%s", text)
		})
	}

	model = &Model{
		options: options,
		cSD:     csd,
	}

	return
}

func (sd *Model) Close() {
	if sd.ctx != nil {
		sd.cSD.FreeSDContext(sd.ctx)
		sd.ctx = nil
	}

	var err = sd.cSD.Close()
	if err != nil {
		println(err.Error())
		return
	}

	return
}

func (sd *Model) LoadFromFile(path string) (err error) {
	if sd.ctx != nil {
		if sd.ctx.Path == path {
			return
		}

		sd.cSD.FreeSDContext(sd.ctx)
		sd.ctx = nil
	}

	_, err = os.Stat(path)
	if err != nil {
		err = errors.New("the system cannot find the model file specified")
		return
	}

	var params = &bind.NewSDContextParams{
		ModelPath:             path,
		LoraModelDir:          sd.options.LoraModelDir,
		VaePath:               sd.options.VaePath,
		TAESDPath:             sd.options.TaesdPath,
		VaeDecodeOnly:         sd.options.VaeDecodeOnly,
		VaeTiling:             sd.options.VaeTiling,
		FreeParamsImmediately: sd.options.FreeParamsImmediately,
		NThreads:              sd.options.Threads,
		WType:                 sd.options.WType,
		RngType:               sd.options.RngType,
		Schedule:              sd.options.Schedule,
	}

	sd.ctx = sd.cSD.NewSDContext(params)
	return
}

func (sd *Model) GetSystemInfo() string {
	return sd.cSD.GetSystemInfo()
}

func (sd *Model) Predict(params *opts.Params) (err error) {
	if sd.ctx == nil {
		return errors.New("model not loaded")
	}

	if params == nil {
		return errors.New("params is nil")
	}

	if params.Width%8 != 0 || params.Height%8 != 0 {
		return errors.New("width and height must be multiples of 8")
	}

	var images = sd.cSD.PredictImage(
		sd.ctx,
		params.Prompt,
		params.NegativePrompt,
		params.ClipSkip,
		params.CfgScale,
		params.Width,
		params.Height,
		params.SampleMethod,
		params.SampleSteps,
		params.Seed,
		params.BatchCount,
	)

	var timeSave = time.Now().Unix()
	for i, img := range images {
		var filename = fmt.Sprintf("./output/%d-%d-%d.png", params.Seed, timeSave, i)

		var file *os.File
		file, err = os.Create(filename)
		if err != nil {
			return
		}

		err = imageToWriter(img, params.OutputsImageType, file)
		if err != nil {
			return
		}

		_ = file.Close()
	}

	return
}

func imageToWriter(image image.Image, imageType opts.OutputsImageType, writer io.Writer) (err error) {
	err = errors.New("unknown image type")

	switch imageType {
	case opts.PNG:
		err = png.Encode(writer, image)
	case opts.JPEG:
		err = jpeg.Encode(writer, image, &jpeg.Options{Quality: 100})
	}

	if err != nil {
		return
	}

	return
}

func (sd *Model) LoadUpscaleModel(esrganPath string) {
	if sd.upscalerCtx == nil {
		sd.esrganPath = esrganPath
		sd.upscalerCtx = sd.cSD.NewUpscalerCtx(esrganPath, sd.options.Threads, sd.options.WType)
	}

	if sd.esrganPath != esrganPath {
		if sd.upscalerCtx != nil {
			sd.cSD.FreeUpscalerCtx(sd.upscalerCtx)
		}
		sd.upscalerCtx = sd.cSD.NewUpscalerCtx(esrganPath, sd.options.Threads, sd.options.WType)
	}
}

func (sd *Model) CloseUpscaleModel() {
	if sd.upscalerCtx != nil {
		sd.cSD.FreeUpscalerCtx(sd.upscalerCtx)
	}
}

func (sd *Model) UpscaleImage(reader io.Reader, upscaleFactor uint32, writer io.Writer) (err error) {
	img, err := sd.cSD.UpscaleImage(sd.upscalerCtx, reader, upscaleFactor)
	if err != nil {
		return
	}

	err = imageToWriter(img, opts.PNG, writer)
	if err != nil {
		return
	}

	return
}
