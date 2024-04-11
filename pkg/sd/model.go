package sd

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"os"

	"github.com/ring-c/go-web-diff/pkg/bind"
	"github.com/ring-c/go-web-diff/pkg/opts"
)

var DefaultOptions = &opts.Options{
	Threads:               -1, // auto
	VaeDecodeOnly:         true,
	FreeParamsImmediately: true,
	RngType:               opts.CUDA_RNG,
	Wtype:                 opts.F32,
	Schedule:              opts.DEFAULT,
}

var DefaultFullParams = &opts.FullParams{
	NegativePrompt:   "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
	CfgScale:         7.0,
	Width:            512,
	Height:           512,
	SampleMethod:     opts.EULER_A,
	SampleSteps:      20,
	Strength:         0.4,
	Seed:             42,
	BatchCount:       1,
	OutputsImageType: opts.PNG,
}

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

func (sd *Model) Close() error {
	return sd.cSD.Close()
}

func (sd *Model) LoadFromFile(path string) (err error) {
	if sd.ctx != nil {
		sd.cSD.FreeCtx(sd.ctx)
		sd.ctx = nil
		log.Printf("model already loaded, free old model")
	}

	_, err = os.Stat(path)
	if err != nil {
		err = errors.New("the system cannot find the model file specified")
		return
	}

	var ctx = sd.cSD.NewCtx(
		path,
		sd.options.VaePath,
		sd.options.TaesdPath,
		sd.options.LoraModelDir,
		sd.options.VaeDecodeOnly,
		sd.options.VaeTiling,
		sd.options.FreeParamsImmediately,
		sd.options.Threads,
		sd.options.Wtype,
		sd.options.RngType,
		sd.options.Schedule,
	)

	sd.ctx = ctx
	return
}

func (sd *Model) GetSystemInfo() string {
	return sd.cSD.GetSystemInfo()
}

func (sd *Model) Predict(prompt string, params *opts.FullParams, writer io.Writer) error {
	if sd.ctx == nil {
		return errors.New("model not loaded")
	}

	if params.Width%8 != 0 || params.Height%8 != 0 {
		return errors.New("width and height must be multiples of 8")
	}

	if params == nil {
		return errors.New("params is nil")
	}

	images := sd.cSD.PredictImage(
		sd.ctx,
		prompt,
		params.NegativePrompt,
		params.ClipSkip,
		params.CfgScale,
		params.Width,
		params.Height,
		params.SampleMethod,
		params.SampleSteps,
		params.Seed,
		1,
	)

	if images == nil || len(images) != 1 {
		return errors.New("predict failed")
	}

	var img = images[0]
	outputsImage := bytesToImage(img.Data, int(img.Width), int(img.Height))

	err := imageToWriter(outputsImage, params.OutputsImageType, writer)
	if err != nil {
		return err
	}

	return nil
}

func bytesToImage(byteData []byte, width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

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
	return img
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

/*
func (sd *Model) UpscaleImage(reader io.Reader, esrganPath string, upscaleFactor uint32, writer io.Writer) (err error) {
	if sd.upscalerCtx == nil {
		sd.esrganPath = esrganPath
		sd.upscalerCtx = sd.cSD.NewUpscalerCtx(esrganPath, sd.options.Threads, sd.options.Wtype)
	}

	if sd.esrganPath != esrganPath {
		if sd.upscalerCtx != nil {
			sd.cSD.FreeUpscalerCtx(sd.upscalerCtx)
		}
		sd.upscalerCtx = sd.cSD.NewUpscalerCtx(esrganPath, sd.options.Threads, sd.options.Wtype)
	}

	decode, _, err := image.Decode(reader)
	if err != nil {
		return
	}

	println("UPSCALE")

	img := sd.cSD.UpscaleImage(sd.upscalerCtx, decode, upscaleFactor)

	spew.Dump(img)

	println("BYTES")

	outputsImage := bytesToImage(img.Data, int(img.Width), int(img.Height))

	println("WRITE")

	err = imageToWriter(outputsImage, opts.PNG, writer)
	if err != nil {
		return
	}

	return
}
*/
