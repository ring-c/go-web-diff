package bind

import (
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"

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

func stringToByteArray(in string) *byte {
	data, err := unix.BytePtrFromString(in)

	if err != nil {
		panic(err.Error())
	}

	return data
}

type NewSDContextGoParams struct {
	ModelPath      *byte
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

	NThreads int
	WType    int
	RngType  int
	Schedule int
}

func (c *CStableDiffusionImpl) NewSDContext(params *NewSDContextParams) *CStableDiffusionCtx {
	var paramsToC = NewSDContextGoParams{
		ModelPath:      stringToByteArray(params.ModelPath),
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

		NThreads: int(params.NThreads),
		WType:    int(params.WType),
		RngType:  int(params.RngType),
		Schedule: int(params.Schedule),
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
