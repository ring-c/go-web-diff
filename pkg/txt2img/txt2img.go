package sd

import (
	"fmt"
	"time"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type SDImageT struct {
	// fields
}

type SDCtxT struct {
	SD *struct {
		StackedID bool
		Denoiser  *struct {
			Schedule *struct {
				GetSigmas func(int) []float32
			}
		}
	}
}

func Txt2Img(
	sdCtx *SDCtxT,
	promptCStr, negativePromptCStr string,
	clipSkip int,
	cfgScale float32,
	width, height int,
	sampleMethod opts.SampleMethod,
	sampleSteps int,
	seed int64,
	batchCount int,
	controlCond *SDImageT,
	controlStrength, styleRatio float32,
	normalizeInput bool,
	inputIDImagesPathCStr string,
) *SDImageT {
	fmt.Printf("txt2img %dx%d\n", width, height)
	if sdCtx == nil {
		return nil
	}

	params := struct {
		MemSize   uint64
		MemBuffer interface{}
		NoAlloc   bool
	}{
		MemSize:   10 * 1024 * 1024, // 10 MB
		MemBuffer: nil,
		NoAlloc:   false,
	}
	if sdCtx.SD.StackedID {
		params.MemSize += 10 * 1024 * 1024 // 10 MB
	}
	params.MemSize += uint64(width * height * 3 * 4) // 3 * sizeof(float)
	params.MemSize *= uint64(batchCount)

	workCtx := ggmlInit(params)
	if workCtx == nil {
		fmt.Println("ggml_init() failed")
		return nil
	}

	t0 := time.Now().UnixMilli()

	sigmas := denoiser.GetSigmas(sampleSteps)

	resultImages := generateImage(sdCtx, workCtx, nil,
		promptCStr,
		negativePromptCStr,
		clipSkip, cfgScale,
		width, height,
		sampleMethod, sigmas,
		seed, batchCount,
		controlCond, controlStrength,
		styleRatio, normalizeInput,
		inputIDImagesPathCStr,
	)

	t1 := time.Now().UnixMilli()

	fmt.Printf("txt2img completed in %.2fs\n", float64(t1-t0)/1000)

	return resultImages
}
