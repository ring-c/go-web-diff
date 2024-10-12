package bind

import (
	"golang.org/x/sys/unix"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type NewSDContextParams struct {
	ModelPath     string
	LoraModelDir  string
	VaePath       string
	NThreads      uint8
	WType         opts.WType
	RngType       opts.RNGType
	Schedule      opts.Schedule
	FluxModelPath string
	ClipLPath     string
	T5xxlPath     string

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

func stringToByteArray(in string) *byte {
	data, err := unix.BytePtrFromString(in)

	if err != nil {
		panic(err.Error())
	}

	return data
}

type NewSDContextGoParams struct {
	ModelPath      *byte
	FluxModelPath  *byte
	ClipLPath      *byte
	T5xxlPath      *byte
	VaePath        *byte
	TaesdPath      *byte
	ControlNetPath *byte
	LoraModelDir   *byte
	EmbedDir       *byte
	IDEmbedDir     *byte

	VaeDecodeOnly         bool
	FreeParamsImmediately bool
	KeepClipOnCPU         bool
	KeepControlNetCPU     bool
	KeepVAEOnCPU          bool
	VaeTiling             bool
	ShowDebug             bool

	NThreads uint8
	WType    uint8
	RngType  uint8
	Schedule uint8
}

func (c *CStableDiffusionImpl) NewSDContext(params *NewSDContextParams) *CStableDiffusionCtx {
	var paramsToC = NewSDContextGoParams{
		ModelPath: stringToByteArray(params.ModelPath),

		FluxModelPath: stringToByteArray(params.FluxModelPath),
		ClipLPath:     stringToByteArray(params.ClipLPath),
		T5xxlPath:     stringToByteArray(params.T5xxlPath),

		VaePath:        stringToByteArray(params.VaePath),
		TaesdPath:      stringToByteArray(params.TAESDPath),
		ControlNetPath: stringToByteArray(params.ControlNetPath),
		LoraModelDir:   stringToByteArray(params.LoraModelDir),
		EmbedDir:       stringToByteArray(params.EmbedDir),
		IDEmbedDir:     stringToByteArray(params.IDEmbedDir),

		VaeTiling:             params.VaeTiling,
		FreeParamsImmediately: params.FreeParamsImmediately,
		VaeDecodeOnly:         params.VaeDecodeOnly,
		KeepClipOnCPU:         params.KeepClipOnCpu,
		KeepControlNetCPU:     params.KeepControlNetCpu,
		KeepVAEOnCPU:          params.KeepVaeOnCpu,
		ShowDebug:             false,

		NThreads: params.NThreads,
		WType:    uint8(params.WType),
		RngType:  uint8(params.RngType),
		Schedule: uint8(params.Schedule),
	}

	return &CStableDiffusionCtx{
		Path: params.ModelPath,
		CTX:  c.newSDContext(&paramsToC),
	}
}

func (c *CStableDiffusionImpl) FreeSDContext(ctx *CStableDiffusionCtx) {
	if ctx != nil && ctx.CTX != nil {
		c.freeSDContext(ctx.CTX)
	}
	ctx = nil
}

/*
func (c *CStableDiffusionImpl) NewUpscalerCtx(esrganPath string, nThreads uint8, wType opts.WType) *CUpScalerCtx {
	return &CUpScalerCtx{
		ctx: c.newUpscalerCtx(esrganPath, nThreads, int(wType)),
	}
}

func (c *CStableDiffusionImpl) FreeUpscalerCtx(ctx *CUpScalerCtx) {
	if ctx != nil {
		c.freeUpscalerCtx(ctx.ctx)
	}
	ctx = nil
	runtime.GC()
}
*/
