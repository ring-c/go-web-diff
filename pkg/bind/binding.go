package bind

import (
	"errors"
	"image"
	"io"
	"os"
	"unsafe"

	"github.com/ebitengine/purego"
	_ "github.com/ianlancetaylor/cgosymbolizer"

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

type Image struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    []byte
}

type CStableDiffusionImpl struct {
	libSd       uintptr
	libFilename string

	txt2img          func(ctx unsafe.Pointer, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, seed int64, batchCount int, controlCond unsafe.Pointer, controlStrength float32, styleRatio float32, normalizeInput bool, inputIdImagesPath string) unsafe.Pointer
	sdGetSystemInfo  func() unsafe.Pointer
	sdSetLogCallback func(callback func(level int, text unsafe.Pointer, data unsafe.Pointer) unsafe.Pointer, data unsafe.Pointer)

	// img2img func(ctx unsafe.Pointer, img uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, strength float32, seed int64, batchCount int) uintptr

	newSDContext          func(params unsafe.Pointer) unsafe.Pointer
	newSDContextParams    func(modelPath, loraModelDir, vaePath string, nThreads int16, wType opts.WType, rngType opts.RNGType, schedule opts.Schedule) unsafe.Pointer
	newSDContextParamsSet func()
	freeSDContext         func(ctx unsafe.Pointer)

	newUpscalerCtx  func(esrganPath string, nThreads int16, wtype int) unsafe.Pointer
	freeUpscalerCtx func(ctx unsafe.Pointer)

	Upscale func(ctx unsafe.Pointer, upscaleFactor, width, height, channel uint32, data []byte) unsafe.Pointer
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
	purego.RegisterLibFunc(&impl.newSDContextParams, libSd, "new_sd_ctx_params")
	purego.RegisterLibFunc(&impl.newSDContextParamsSet, libSd, "new_sd_ctx_params_set")

	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")

	purego.RegisterLibFunc(&impl.Upscale, libSd, "upscale_go")

	return &impl, err
}

func (c *CStableDiffusionImpl) PredictImage(ctx *CStableDiffusionCtx, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod opts.SampleMethod, sampleSteps int, seed int64) (result image.Image) {
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

func (c *CStableDiffusionImpl) UpscaleImage(ctx *CUpScalerCtx, reader io.Reader, upscaleFactor uint32) (result image.Image, err error) {
	decode, _, err := image.Decode(reader)
	if err != nil {
		return
	}

	var img = imageToBytes(decode)

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
