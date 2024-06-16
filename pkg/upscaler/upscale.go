package upscaler

import (
	"errors"
	"fmt"
	"image"
)

func (u *Upscaler) upscale(inputImage image.Image) (upscaledImage image.Image, err error) {
	inputWidth := inputImage.Bounds().Max.X
	inputHeight := inputImage.Bounds().Max.Y

	outputWidth := inputWidth * 4
	outputHeight := inputHeight * 4
	fmt.Printf(
		"upscaling from (%d x %d) to (%d x %d)\n",
		inputWidth, inputHeight,
		outputWidth, outputHeight,
	)

	var memSize = uint64(outputWidth * outputHeight * 3 * 4 * 2)

	u.upscaleCTX = u.GGML.InitGo(memSize)
	if u.upscaleCTX == nil {
		err = errors.New("GGML.InitGo failed")
		return
	}
	fmt.Printf("upscale work buffer size: %.2f MB\n", float64(memSize)/1024/1024)

	var input = u.GGML.NewTensor4D(u.upscaleCTX, 0, inputWidth, inputHeight, 3, 1)
	u.imageToTensor(inputImage, input)

	/*


		upscaled := ggmlNewTensor4d(upscaleCtx, GGML_TYPE_F32, uint32(outputWidth), uint32(outputHeight), 3, 1)
		onTiling := func(in, out *ggmlTensor, init bool) {
			esrganUpscaler.compute(nThreads, in, out)
		}
		t0 := ggmlTimeMs()
		u.sdTiling(inputImageTensor, upscaled, esrganUpscaler.scale, esrganUpscaler.tileSize, 0.25, onTiling)
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
	*/
	return
}
