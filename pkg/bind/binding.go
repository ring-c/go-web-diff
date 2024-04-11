package bind

import (
	"os"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type CStableDiffusionCtx struct {
	ctx unsafe.Pointer
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

type cDarwinImage struct {
	width   uint32
	height  uint32
	channel uint32
	data    *byte
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

	txt2img         func(ctx unsafe.Pointer, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, seed int64, batchCount int) unsafe.Pointer
	sdGetSystemInfo func() unsafe.Pointer

	newSdCtx func(modelPath string, vaePath string, taesdPath string, loraModelDir string, vaeDecodeOnly bool, vaeTiling bool, freeParamsImmediately bool, nThreads int, wType int, rngType int, schedule int) unsafe.Pointer

	sdSetLogCallback func(callback func(level int, text unsafe.Pointer, data unsafe.Pointer) unsafe.Pointer, data unsafe.Pointer)
	img2img          func(ctx unsafe.Pointer, img uintptr, prompt string, negativePrompt string, clipSkip int, cfgScale float32, width int, height int, sampleMethod int, sampleSteps int, strength float32, seed int64, batchCount int) uintptr
	freeSdCtx        func(ctx unsafe.Pointer)
	newUpscalerCtx   func(esrganPath string, nThreads int, wtype int) unsafe.Pointer
	freeUpscalerCtx  func(ctx unsafe.Pointer)
	upscale          func(ctx unsafe.Pointer, img unsafe.Pointer, upscaleFactor uint32) uintptr
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
	purego.RegisterLibFunc(&impl.newSdCtx, libSd, "new_sd_ctx")
	purego.RegisterLibFunc(&impl.sdSetLogCallback, libSd, "sd_set_log_callback")
	purego.RegisterLibFunc(&impl.img2img, libSd, "img2img")
	purego.RegisterLibFunc(&impl.freeSdCtx, libSd, "free_sd_ctx")
	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")
	purego.RegisterLibFunc(&impl.upscale, libSd, "upscale")

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

/*

func (c *CStableDiffusionImpl) NewUpscalerCtx(esrganPath string, nThreads int, wType opts.WType) *CUpScalerCtx {
	ctx := c.newUpscalerCtx(esrganPath, nThreads, int(wType))

	return &CUpScalerCtx{ctx: ctx}
}

func (c *CStableDiffusionImpl) FreeUpscalerCtx(ctx *CUpScalerCtx) {
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&ctx.ctx))
	if ptr != nil {
		c.freeUpscalerCtx(ctx.ctx)
	}
	ctx = nil
	runtime.GC()
}

func (c *CStableDiffusionImpl) UpscaleImage(ctx *CUpScalerCtx, image image.Image, upscaleFactor uint32) Image {
	// img := imageToBytes(image)
	//
	// var ci = cImage{
	// 	width:   img.Width,
	// 	height:  img.Height,
	// 	channel: img.Channel,
	// 	data:    unsafe.Pointer(&img.Data[0]),
	// }

	println("TEAPOT 1")

	uPtr := c.upscale(ctx.ctx, nil, upscaleFactor)

	println("TEAPOT 2")

	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&uPtr))
	if ptr == nil {
		return Image{}
	}
	println("TEAPOT 3")

	cimg := (*cImage)(ptr)
	dataPtr := *(*unsafe.Pointer)(unsafe.Pointer(&cimg.data))
	return Image{
		Width:   cimg.width,
		Height:  cimg.height,
		Channel: cimg.channel,
		Data:    unsafe.Slice((*byte)(dataPtr), cimg.channel*cimg.width*cimg.height),
	}
}
*/
