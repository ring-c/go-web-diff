package sd

func computeFirstStage(workCtx *ggml.Context, x *ggml.Tensor, decode bool) *ggml.Tensor {
	W := x.Ne[0]
	H := x.Ne[1]
	var result *ggml.Tensor
	if decode {
		result = ggml.NewTensor4D(workCtx, ggml.TYPE_F32, W*8, H*8, 3, x.Ne[3])
	} else {
		result = ggml.NewTensor4D(workCtx, ggml.TYPE_F32, W/8, H/8, useTinyAutoencoder ? 4 : 8, x.Ne[3])
	}
	t0 := ggml.TimeMs()
	if !useTinyAutoencoder {
		if decode {
			ggml.TensorScale(x, 1.0/scaleFactor)
		} else {
			ggml.TensorScaleInput(x)
		}
		if vaeTiling && decode {
			// TODO: support tiling vae encode
			onTiling := func(in, out *ggml.Tensor, init bool) {
				firstStageModel.Compute(nThreads, in, decode, &out)
			}
			sdTiling(x, result, 8, 32, 0.5, onTiling)
		} else {
			firstStageModel.Compute(nThreads, x, decode, &result)
		}
		firstStageModel.FreeComputeBuffer()
		if decode {
			ggml.TensorScaleOutput(result)
		}
	} else {
		if vaeTiling && decode {
			// TODO: support tiling vae encode
			onTiling := func(in, out *ggml.Tensor, init bool) {
				taeFirstStage.Compute(nThreads, in, decode, &out)
			}
			sdTiling(x, result, 8, 64, 0.5, onTiling)
		} else {
			taeFirstStage.Compute(nThreads, x, decode, &result)
		}
		taeFirstStage.FreeComputeBuffer()
	}
	t1 := ggml.TimeMs()
	logDebug("computing vae [mode: %s] graph completed, taking %.2fs", map[bool]string{true: "DECODE", false: "ENCODE"}[decode], float32(t1-t0)/1000)
	if decode {
		ggml.TensorClamp(result, 0.0, 1.0)
	}
	return result
}

