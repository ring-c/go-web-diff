package upscaler

func upscale(inputImage sdImageT, upscaleFactor uint32) sdImageT {
	// upscaleFactor is unused for RealESRGAN_x4plus_anime_6B.pth
	var upscaledImage sdImageT

	outputWidth := int(inputImage.width) * esrganUpscaler.scale
	outputHeight := int(inputImage.height) * esrganUpscaler.scale
	logInfo("upscaling from (%d x %d) to (%d x %d)", inputImage.width, inputImage.height, outputWidth, outputHeight)

	params := ggmlInitParams{
		memSize:   uint64(outputWidth * outputHeight * 3 * 4 * 2),
		memBuffer: nil,
		noAlloc:   false,
	}
	params.memSize += 2 * ggmlTensorOverhead()

	// draft context
	upscaleCtx := ggmlInit(params)
	if upscaleCtx == nil {
		logError("ggml_init() failed")
		return upscaledImage
	}
	logDebug("upscale work buffer size: %.2f MB", float64(params.memSize)/1024/1024)

	inputImageTensor := ggmlNewTensor4d(upscaleCtx, GGML_TYPE_F32, inputImage.width, inputImage.height, 3, 1)
	sdImageToTensor(inputImage.data, inputImageTensor)

	upscaled := ggmlNewTensor4d(upscaleCtx, GGML_TYPE_F32, uint32(outputWidth), uint32(outputHeight), 3, 1)
	onTiling := func(in, out *ggmlTensor, init bool) {
		esrganUpscaler.compute(nThreads, in, out)
	}
	t0 := ggmlTimeMs()
	sdTiling(inputImageTensor, upscaled, esrganUpscaler.scale, esrganUpscaler.tileSize, 0.25, onTiling)
	esrganUpscaler.freeComputeBuffer()
	ggmlTensorClamp(upscaled, 0, 1)
	upscaledData := sdTensorToImage(upscaled)
	ggmlFree(upscaleCtx)
	t3 := ggmlTimeMs()
	logInfo("input_image_tensor upscaled, taking %.2fs", float64(t3-t0)/1000)

	upscaledImage = sdImageT{
		width:  uint32(outputWidth),
		height: uint32(outputHeight),
		depth:  3,
		data:   upscaledData,
	}
	return upscaledImage
}
