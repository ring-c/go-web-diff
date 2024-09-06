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

	// sdSetLogCallback func(callback func(level int, text uintptr, data uintptr) uintptr, data int)

	newSDContext  func(params *NewSDContextGoParams) unsafe.Pointer
	freeSDContext func(ctx unsafe.Pointer)

	newUpscalerCtx  func(esrganPath string, nThreads int16, wtype int) unsafe.Pointer
	freeUpscalerCtx func(ctx unsafe.Pointer)

	Text2Image func(
		sdCTX unsafe.Pointer,
		prompt, negPrompt string,
		clipSkip int,
		cfgScale, guidance float32,
		width, height int,
		sampleMethod int,
		sampleSteps int,
		seed int64,
		batchCount int,
		controlCond unsafe.Pointer,
		controlStrength, styleStrength float32,
		normalizeInput bool,
		inputIDImagesPath string,
	) unsafe.Pointer

	Upscale func(ctx unsafe.Pointer, upscaleFactor, width, height, channel uint32, data []byte) unsafe.Pointer
}

func NewCStableDiffusion() (*CStableDiffusionImpl, error) {
	libSd, filename, err := OpenLibrary()
	if err != nil {
		return nil, err
	}

	var impl = CStableDiffusionImpl{
		libSd:       libSd,
		libFilename: filename,
	}

	// purego.RegisterLibFunc(&impl.sdSetLogCallback, libSd, "sd_set_log_callback")
	purego.RegisterLibFunc(&impl.newSDContext, libSd, "new_sd_ctx_go")
	purego.RegisterLibFunc(&impl.freeSDContext, libSd, "free_sd_ctx")

	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")

	purego.RegisterLibFunc(&impl.Upscale, libSd, "upscale_go")
	purego.RegisterLibFunc(&impl.Text2Image, libSd, "txt2img")

	return &impl, err
}

func (c *CStableDiffusionImpl) Close() {
	if c.libSd != 0 {
		_ = CloseLibrary(c.libSd)
	}

	if len(c.libFilename) > 0 {
		_ = os.Remove(c.libFilename)
	}
}

func (c *CStableDiffusionImpl) UpscaleImage(ctx *CUpScalerCtx, decoded image.Image, upscaleFactor uint32) (result *image.RGBA, err error) {
	var img = ImageToBytes(decoded)

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

/*

func (c *CStableDiffusionImpl) SetLogCallBack() {

	var cb = func(level int, text uintptr, data uintptr) uintptr {
		// spew.Dump(goString(text))
		println(text)
		return 0
	}

	c.sdSetLogCallback(cb, 0)
}

func goString(c uintptr) string {
	// We take the address and then dereference it to trick go vet from creating a possible misuse of unsafe.Pointer
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&c))
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Add(ptr, uintptr(length))) == '\x00' {
			break
		}
		length++
	}
	return unsafe.String((*byte)(ptr), length)
}
*/
