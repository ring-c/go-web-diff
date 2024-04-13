package bind

import (
	"runtime"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (c *CStableDiffusionImpl) NewCtx(modelPath string, vaePath string, controlNetPath string, taesdPath string, loraModelDir string, embedDir string, idEmbedDir string, vaeDecodeOnly bool, vaeTiling bool, freeParamsImmediately bool, nThreads int, wType opts.WType, rngType opts.RNGType, schedule opts.Schedule, keepClipOnCpu bool, keepControlNetCpu bool, keepVaeOnCpu bool) *CStableDiffusionCtx {
	return &CStableDiffusionCtx{
		ctx: c.newSdCtx(modelPath),
	}
}

func (c *CStableDiffusionImpl) FreeCtx(ctx *CStableDiffusionCtx) {
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&ctx.ctx))
	if ptr != nil {
		c.freeSdCtx(ctx.ctx)
	}
	ctx = nil
	runtime.GC()
}

func (c *CStableDiffusionImpl) NewUpscalerCtx(esrganPath string, nThreads int, wType opts.WType) *CUpScalerCtx {
	return &CUpScalerCtx{
		ctx: c.newUpscalerCtx(esrganPath, nThreads, int(wType)),
	}
}

func (c *CStableDiffusionImpl) FreeUpscalerCtx(ctx *CUpScalerCtx) {
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&ctx.ctx))
	if ptr != nil {
		c.freeUpscalerCtx(ctx.ctx)
	}
	ctx = nil
	runtime.GC()
}
