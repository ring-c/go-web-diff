package sd

import (
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"sync"
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
			print(text)
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

func (sd *Model) Predict(params *opts.Params, debug bool) (filenames []string, err error) {
	filenames = make([]string, 0)
	if sd.ctx == nil {
		err = errors.New("model not loaded")
		return
	}

	if params == nil {
		err = errors.New("params is nil")
		return
	}

	if params.Width%8 != 0 || params.Height%8 != 0 {
		err = errors.New("width and height must be multiples of 8")
		return
	}

	var seed = params.Seed
	if seed == -1 {
		seed = rand.Int63()
	}

	var timeSave = time.Now().Unix()
	for i := 0; i < params.BatchCount; i++ {
		if debug {
			fmt.Printf("\nGenerating for seed %d\n\n", seed)
		}

		var data = sd.cSD.PredictImage(
			sd.ctx,
			params.Prompt,
			params.NegativePrompt,
			params.ClipSkip,
			params.CfgScale,
			params.Width,
			params.Height,
			params.SampleMethod,
			params.SampleSteps,
			seed,
		)

		if data.Bounds().Max.X == 0 || data.Bounds().Max.Y == 0 {
			println("err generate, size 0x0")
			continue
		}

		var filename = fmt.Sprintf("%d-%d-%d.png", timeSave, seed, i)
		var file *os.File
		file, err = os.Create("./output/" + filename)
		if err != nil {
			return
		}

		err = imageToWriter(data, file)
		if err != nil {
			return
		}

		err = file.Close()
		if err != nil {
			return
		}

		filenames = append(filenames, filename)
		seed++
	}

	return
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

func (sd *Model) LoadUpscaleModel(esrganPath string) (err error) {
	if _, err = os.Stat(esrganPath); errors.Is(err, os.ErrNotExist) {
		err = errors.New("upscale model does not exists")
		return
	}

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

	return
}

func (sd *Model) CloseUpscaleModel() {
	if sd.upscalerCtx != nil {
		sd.cSD.FreeUpscalerCtx(sd.upscalerCtx)
	}
}

func (sd *Model) UpscaleImage(wg *sync.WaitGroup, filenameIn, filenameOut string, upscaleFactor uint32) (err error) {
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

	data, err := sd.cSD.UpscaleImage(sd.upscalerCtx, decoded, upscaleFactor)
	if err != nil {
		return
	}

	if data.Bounds().Max.X == 0 || data.Bounds().Max.Y == 0 {
		err = errors.New("size 0x0")
		return
	}

	wg.Add(1)
	go func(wgg *sync.WaitGroup, img *image.RGBA, filename string) {
		defer wgg.Done()

		var fileWrite, err = os.Create(filename)
		if err != nil {
			println(err.Error())
			return
		}

		defer func() {
			_ = fileWrite.Close()
		}()

		err = imageToWriter(img, fileWrite)
		if err != nil {
			println(err.Error())
			return
		}
	}(wg, data, filenameOut)

	return
}
