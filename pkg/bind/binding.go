package bind

import (
	"errors"
	"io"
	"os"
	"unsafe"

	"github.com/ebitengine/purego"
	_ "github.com/ianlancetaylor/cgosymbolizer"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type CStableDiffusionCtx struct {
	ctx uintptr
}

type CUpScalerCtx struct {
	ctx uintptr
}

type CLogCallback func(level opts.LogLevel, text string)

type cImage struct {
	width   uint32
	height  uint32
	channel uint32
	data    uintptr
}

type Image struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    []byte
}

type CStableDiffusionImpl struct {
	libSd       uintptr
	libFilename string

	txt2img          func(ctx uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, seed int64, batchCount int, controlCond unsafe.Pointer, controlStrength float32, styleRatio float32, normalizeInput bool, inputIdImagesPath string) unsafe.Pointer
	sdGetSystemInfo  func() unsafe.Pointer
	sdSetLogCallback func(callback func(level int, text unsafe.Pointer, data unsafe.Pointer) unsafe.Pointer, data unsafe.Pointer)

	img2img func(ctx unsafe.Pointer, img uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, strength float32, seed int64, batchCount int) uintptr
	upscale func(ctx *CUpScalerCtx, img unsafe.Pointer, upscaleFactor uint32) unsafe.Pointer

	newSdCtx  func(modelPath string) uintptr
	freeSdCtx func(ctx uintptr)

	newUpscalerCtx  func(esrganPath string, nThreads int, wtype int) uintptr
	freeUpscalerCtx func(ctx uintptr)

	newSDImage func() uintptr
	Generate   func(ctx uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, seed int64, batchCount int, withUpscale bool, upscaleScale int) unsafe.Pointer
}

func NewCStableDiffusion() (*CStableDiffusionImpl, error) {
	libSd, filename, err := openLibrary()
	if err != nil {
		return nil, err
	}

	var impl = CStableDiffusionImpl{
		libSd:       libSd,
		libFilename: filename,
	}

	purego.RegisterLibFunc(&impl.txt2img, libSd, "txt2img")
	purego.RegisterLibFunc(&impl.sdGetSystemInfo, libSd, "sd_get_system_info")
	purego.RegisterLibFunc(&impl.sdSetLogCallback, libSd, "sd_set_log_callback")

	purego.RegisterLibFunc(&impl.img2img, libSd, "img2img")
	purego.RegisterLibFunc(&impl.upscale, libSd, "upscale")

	purego.RegisterLibFunc(&impl.newSdCtx, libSd, "new_sd_ctx_go")
	purego.RegisterLibFunc(&impl.freeSdCtx, libSd, "free_sd_ctx")

	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")

	purego.RegisterLibFunc(&impl.Generate, libSd, "generate")

	return &impl, err
}

func (c *CStableDiffusionImpl) PredictImage(ctx *CStableDiffusionCtx, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod opts.SampleMethod, sampleSteps int, seed int64, batchCount int) []Image {
	images := c.txt2img(
		ctx.ctx,
		prompt,
		negativePrompt,
		clipSkip,
		cfgScale,
		width,
		height,
		int(sampleMethod),
		sampleSteps,
		seed,
		batchCount,
		nil,
		0,
		0,
		false,
		"",
	)

	return goImageSlice(images, batchCount)
}

func (c *CStableDiffusionImpl) SetLogCallBack(cb CLogCallback) {
	c.sdSetLogCallback(func(level int, text unsafe.Pointer, data unsafe.Pointer) unsafe.Pointer {
		cb(opts.LogLevel(level), goString(text))
		return nil
	}, nil)
}

func (c *CStableDiffusionImpl) Close() (err error) {
	if c.libSd != 0 {
		err = closeLibrary(c.libSd)
		if err != nil {
			return
		}
	}

	if len(c.libFilename) > 0 {
		err = os.Remove(c.libFilename)
		if err != nil {
			return
		}
	}

	return
}

func (c *CStableDiffusionImpl) GetSystemInfo() string {
	return goString(c.sdGetSystemInfo())
}

func (c *CStableDiffusionImpl) UpscaleImage(ctxUp *CUpScalerCtx, reader io.Reader, upscaleFactor uint32, ctxSD *CStableDiffusionCtx) (result Image, err error) {
	// decode, _, err := image.Decode(reader)
	// if err != nil {
	// 	return
	// }
	//
	// var img = imageToBytes(decode)

	println("TEST 1")

	var params = &opts.FullParams{
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

	params.Width = 256
	params.Height = 256
	params.CfgScale = 2
	params.SampleSteps = 32
	params.SampleMethod = opts.EULER_A
	params.Seed = 4242

	var newSDImage = c.Generate(
		ctxSD.ctx,
		"1girl, indoors, full body",
		params.NegativePrompt,
		params.ClipSkip,
		params.CfgScale,
		params.Width,
		params.Height,
		0,
		params.SampleSteps,
		params.Seed,
		1,
		false,
		2,
	)

	// var newSDImage = c.newSDImage()

	println("TEST 2")
	/*

		// var ci = &cImage{
		// 	width:   img.Width,
		// 	height:  img.Height,
		// 	channel: img.Channel,
		// 	data:    uintptr(unsafe.Pointer(&img.Data[0])),
		// }

		uPtr := c.upscale(ctx, newSDImage, upscaleFactor)
		println("TEST 3")
		spew.Dump(uPtr)

	*/

	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&newSDImage))
	if ptr == nil {
		err = errors.New("nil pointer")
		return
	}

	cimg := (*cImage)(ptr)
	dataPtr := *(*unsafe.Pointer)(unsafe.Pointer(&cimg.data))

	result = Image{
		Width:   cimg.width,
		Height:  cimg.height,
		Channel: cimg.channel,
		Data:    unsafe.Slice((*byte)(dataPtr), cimg.channel*cimg.width*cimg.height),
	}

	return
}
