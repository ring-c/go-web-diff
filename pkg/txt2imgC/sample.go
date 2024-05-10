package sd

func sample(
	workCtx *ggml.Context,
	xT *ggml.Tensor,
	noise *ggml.Tensor,
	c *ggml.Tensor,
	cConcat *ggml.Tensor,
	cVector *ggml.Tensor,
	uc *ggml.Tensor,
	ucConcat *ggml.Tensor,
	ucVector *ggml.Tensor,
	controlHint *ggml.Tensor,
	controlStrength float32,
	minCfg float32,
	cfgScale float32,
	method sampleMethodT,
	sigmas []float64,
	startMergeStep int,
	cID *ggml.Tensor,
	cVecID *ggml.Tensor,
) *ggml.Tensor {
	steps := len(sigmas) - 1
	x := ggml.DupTensor(workCtx, xT)
	ggml.CopyTensor(x, xT)

	noisedInput := ggml.DupTensor(workCtx, xT)

	hasUnconditioned := cfgScale != 1.0 && uc != nil

	if noise == nil {
		ggml.TensorScale(x, sigmas[0])
	} else {
		ggml.TensorScale(noise, sigmas[0])
		ggml.TensorAdd(x, noise)
	}

	outCond := ggml.DupTensor(workCtx, x)
	var outUncond *ggml.Tensor
	if hasUnconditioned {
		outUncond = ggml.DupTensor(workCtx, x)
	}
	denoised := ggml.DupTensor(workCtx, x)

	SamplerEulerA(workCtx)

	if controlNet != nil {
		controlNet.FreeControlCtx()
		controlNet.FreeComputeBuffer()
	}
	diffusionModel.FreeComputeBuffer()
	return x
}
