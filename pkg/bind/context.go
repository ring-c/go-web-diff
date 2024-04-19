package bind

import (
	"runtime"
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type NewSDContextParams struct {
	ModelPath    string
	LoraModelDir string
	VaePath      string
	NThreads     int16
	WType        opts.WType
	RngType      opts.RNGType
	Schedule     opts.Schedule

	// newSDContextParamsSet
	TAESDPath             string
	ControlNetPath        string
	EmbedDir              string
	IDEmbedDir            string
	VaeDecodeOnly         bool
	VaeTiling             bool
	FreeParamsImmediately bool
	KeepClipOnCpu         bool
	KeepControlNetCpu     bool
	KeepVaeOnCpu          bool
}

func (c *CStableDiffusionImpl) NewSDContext(params *NewSDContextParams) *CStableDiffusionCtx {
	var paramsC = c.newSDContextParams(
		params.ModelPath,
		params.LoraModelDir,
		params.VaePath,
		params.NThreads,
		params.WType,
		params.RngType,
		params.Schedule,
	)

	return &CStableDiffusionCtx{
		Path: params.ModelPath,
		CTX:  c.newSDContext(paramsC),
	}
}

func (c *CStableDiffusionImpl) FreeSDContext(ctx *CStableDiffusionCtx) {
	if ctx != nil && ctx.CTX != nil {
		c.freeSDContext(ctx.CTX)
	}
	ctx = nil
}

func (c *CStableDiffusionImpl) NewUpscalerCtx(esrganPath string, nThreads int16, wType opts.WType) *CUpScalerCtx {
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
