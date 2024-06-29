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

	newSDContext  func(params *NewSDContextGoParams) unsafe.Pointer
	freeSDContext func(ctx unsafe.Pointer)

	newUpscalerCtx  func(esrganPath string, nThreads int16, wtype int) unsafe.Pointer
	freeUpscalerCtx func(ctx unsafe.Pointer)

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

	// purego.RegisterLibFunc(&impl.img2img, libSd, "img2img")

	purego.RegisterLibFunc(&impl.newSDContext, libSd, "new_sd_ctx_go")
	purego.RegisterLibFunc(&impl.freeSDContext, libSd, "free_sd_ctx")

	purego.RegisterLibFunc(&impl.newUpscalerCtx, libSd, "new_upscaler_ctx")
	purego.RegisterLibFunc(&impl.freeUpscalerCtx, libSd, "free_upscaler_ctx")

	purego.RegisterLibFunc(&impl.Upscale, libSd, "upscale_go")

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
