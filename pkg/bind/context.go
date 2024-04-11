package bind

import (
	"runtime"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (c *CStableDiffusionImpl) NewCtx(modelPath string, vaePath string, taesdPath string, loraModelDir string, vaeDecodeOnly bool, vaeTiling bool, freeParamsImmediately bool, nThreads int, wType opts.WType, rngType opts.RNGType, schedule opts.Schedule) *CStableDiffusionCtx {
	ctx := c.newSdCtx(modelPath, vaePath, taesdPath, loraModelDir, vaeDecodeOnly, vaeTiling, freeParamsImmediately, nThreads, int(wType), int(rngType), int(schedule))
	return &CStableDiffusionCtx{
		ctx: ctx,
	}
}

func (c *CStableDiffusionImpl) FreeCtx(ctx *CStableDiffusionCtx) {
	ptr := *(*unsafe.Pointer)(&ctx.ctx)
	if ptr != nil {
		c.freeSdCtx(ctx.ctx)
	}
	ctx = nil
	runtime.GC()
}
