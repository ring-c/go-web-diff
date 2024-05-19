package bind

import (
	"errors"
	"image"
	"os"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type CStableDiffusionCtx struct {
	Path string
	CTX  unsafe.Pointer
}

type CUpScalerCtx struct {
	ctx unsafe.Pointer
}

type CLogCallback func(level opts.LogLevel, text string)

type cImage struct {
	width   uint32
	height  uint32
	channel uint32
	data    unsafe.Pointer
}

type CStableDiffusionImpl struct {
	libSd       uintptr
	libFilename string

	txt2img func(
		ctx unsafe.Pointer,
		prompt string,
		negativePrompt string,
		clipSkip int,
		cfgScale float32,
		width int,
		height int,
		sampleMethod int,
		sampleSteps int,
		seed int64,
		batchCount int,
		controlCond unsafe.Pointer,
		controlStrength float32,
		styleRatio float32,
		normalizeInput bool,
		inputIdImagesPath string,
	) unsafe.Pointer

	sdGetSystemInfo  func() unsafe.Pointer
	sdSetLogCallback func(callback func(level int, text unsafe.Pointer, data unsafe.Pointer) unsafe.Pointer, data unsafe.Pointer)

	// img2img func(ctx unsafe.Pointer, img uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, strength float32, seed int64, batchCount int) uintptr

	newSDContext  func(params *NewSDContextGoParams) unsafe.Pointer
	freeSDContext func(ctx unsafe.Pointer)

	newUpscalerCtx  func(esrganPath string, nThreads int16, wtype int) unsafe.Pointer
	freeUpscalerCtx func(ctx unsafe.Pointer)

	Upscale func(ctx unsafe.Pointer, upscaleFactor, width, height, channel uint32, data []byte) unsafe.Pointer

	Test uintptr
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

	// purego.RegisterLibFunc(&impl.img2img, libSd, "img2img")

	purego.RegisterLibFunc(&impl.newSDContext, libSd, "new_sd_ctx_go")
	purego.RegisterLibFunc(&impl.freeSDContext, libSd, "free_sd_ctx")

	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")

	purego.RegisterLibFunc(&impl.Upscale, libSd, "upscale_go")

	return &impl, err
}

func (c *CStableDiffusionImpl) PredictImage(ctx *CStableDiffusionCtx, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod opts.SampleMethod, sampleSteps int, seed int64) (result *image.RGBA) {
	var cImages = c.txt2img(
		ctx.CTX,
		prompt,
		negativePrompt,
		clipSkip,
		cfgScale,
		width,
		height,
		int(sampleMethod),
		sampleSteps,
		seed,
		1,
		nil,
		0,
		0,
		false,
		"",
	)

	var img = goImageSlice(cImages, 1)
	return bytesToImage(img[0].Data, int(img[0].Width), int(img[0].Height))
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
		_ = os.Remove(c.libFilename)
	}

	return
}

func (c *CStableDiffusionImpl) GetSystemInfo() string {
	return goString(c.sdGetSystemInfo())
}

func (c *CStableDiffusionImpl) UpscaleImage(ctx *CUpScalerCtx, decoded image.Image, upscaleFactor uint32) (result *image.RGBA, err error) {
	var img = imageToBytes(decoded)

	var newSDImage = c.Upscale(
		ctx.ctx,
		upscaleFactor,
		img.Width,
		img.Height,
		img.Channel,
		img.Data,
	)

	if newSDImage == nil {
		err = errors.New("nil pointer")
		return
	}

	var cImg = (*cImage)(newSDImage)

	result = bytesToImage(
		unsafe.Slice((*byte)(cImg.data), cImg.channel*cImg.width*cImg.height),
		int(cImg.width),
		int(cImg.height),
	)

	return
}
