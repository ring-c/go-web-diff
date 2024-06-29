package sd

import (
	"errors"
	"image"
	"image/png"
	"io"
	"os"
	"sync"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/bind"
	"github.com/ring-c/go-web-diff/pkg/opts"
)

type Model struct {
	options *opts.Options

	cSD *bind.CStableDiffusionImpl
	ctx *bind.CStableDiffusionCtx

	esrganPath  string
	upscalerCtx *bind.CUpScalerCtx
}

func NewModel(in *opts.Options) (model *Model, err error) {
	csd, err := bind.NewCStableDiffusion()
	if err != nil {
		return
	}

	model = &Model{
		options: in,
		cSD:     csd,
	}

	return
}

func (sd *Model) Close() {
	if sd.ctx != nil {
		sd.cSD.FreeSDContext(sd.ctx)
		sd.ctx = nil
	}

	sd.cSD.Close()
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
	if sd.ctx.CTX == nil {
		err = errors.New("error sd context creation")
		return
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

func (sd *Model) GetCTX() unsafe.Pointer {
	return sd.ctx.CTX
}
