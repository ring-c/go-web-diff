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

func stringToByteArray(in string) *[]byte {
	var data = []byte(in)
	return &data
}

func (c *CStableDiffusionImpl) NewSDContext(params *NewSDContextParams) *CStableDiffusionCtx {

	/*
		// typedef struct {
		//  const char *model_path;
		//  const char *vae_path;
		//  const char *taesd_path;
		//  const char *control_net_path;
		//  const char *lora_model_dir;
		//  const char *embed_dir;
		//  const char *id_embed_dir;
		//  bool vae_decode_only;
		//  bool vae_tiling;
		//  bool free_params_immediately;
		//  int n_threads;
		//  enum sd_type_t wType;
		//  enum rng_type_t rng_type;
		//  enum schedule_t schedule;
		//  bool keep_clip_on_cpu;
		//  bool keep_control_net_cpu;
		//  bool keep_vae_on_cpu;
		// } new_sd_ctx_go_params;

		type NewSDContextGoParams struct {
			ModelPath             *[]byte
			VaePath               *[]byte
			TaesdPath             *[]byte
			ControlNetPath        *[]byte
			LoraModelDir          *[]byte
			EmbedDir              *[]byte
			IDEmbedDir            *[]byte
			VaeDecodeOnly         bool
			VaeTiling             bool
			FreeParamsImmediately bool
			NThreads              int16
			WType                 opts.WType
			RngType               opts.RNGType
			Schedule              opts.Schedule
			KeepClipOnCPU         bool
			KeepControlNetCPU     bool
			KeepVAEOnCPU          bool
		}

		spew.Dump(params)

		var paramsToC = NewSDContextGoParams{
			ModelPath:      stringToByteArray(params.ModelPath),
			LoraModelDir:   stringToByteArray(params.LoraModelDir),
			VaePath:        stringToByteArray(params.VaePath),
			TaesdPath:      stringToByteArray(params.TAESDPath),
			ControlNetPath: stringToByteArray(params.ControlNetPath),
			EmbedDir:       stringToByteArray(params.EmbedDir),
			IDEmbedDir:     stringToByteArray(params.IDEmbedDir),

			NThreads: params.NThreads,
			WType:    params.WType,
			RngType:  params.RngType,
			Schedule: params.Schedule,
		}

		var TypeNewSdCtxGoParams = ffi.Type{
			Type: ffi.Struct,
			Elements: &[]*ffi.Type{
				&ffi.TypePointer, // model_path
				&ffi.TypePointer, // vae_path
				// &ffi.TypePointer, // taesd_path
				// &ffi.TypePointer, // control_net_path
				// &ffi.TypePointer, // lora_model_dir
				// &ffi.TypePointer, // embed_dir
				// &ffi.TypePointer, // id_embed_dir

				// &ffi.TypeUint32, // vae_decode_only
				// &ffi.TypeUint32, // vae_tiling
				// &ffi.TypeUint32, // free_params_immediately
				//
				// &ffi.TypeSint32, // n_threads
				// &ffi.TypeSint32, // wType
				// &ffi.TypeSint32, // rng_type
				// &ffi.TypeSint32, // schedule
				//
				// &ffi.TypeUint32, // keep_clip_on_cpu
				// &ffi.TypeUint32, // keep_control_net_cpu
				// &ffi.TypeUint32, // keep_vae_on_cpu
			}[0],
		}

		var cifNewCTX ffi.Cif
		if ok := ffi.PrepCif(&cifNewCTX, ffi.DefaultAbi, 1, &TypeNewSdCtxGoParams, &ffi.TypePointer); ok != ffi.OK {
			panic("prep failed")
		}

		var fn = func(params NewSDContextGoParams) unsafe.Pointer {
			var result unsafe.Pointer
			ffi.Call(&cifNewCTX, c.Test, unsafe.Pointer(&result), unsafe.Pointer(&params))
			return result
		}

		return &CStableDiffusionCtx{
			Path: params.ModelPath,
			CTX:  fn(paramsToC),
		}
	*/

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
