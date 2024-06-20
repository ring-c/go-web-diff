package txt2img

import (
	"bytes"
	"fmt"
	"math"
	"time"
)

type ggmlContext struct{}
type ggmlTensor struct {
	data []float32
	ne   [3]int
}

var (
	VERSION_XL       = 1
	EOS_TOKEN_ID     = 0
	n_threads        = 4
	cond_stage_model = &CondStageModel{}
	diffusion_model  = &DiffusionModel{}
	version          = VERSION_XL
)

type CondStageModel struct{}

func (c *CondStageModel) tokenize(text string, flag bool) ([]int, []float32) {
	// Dummy implementation
	return []int{1, 2, 3}, []float32{0.1, 0.2, 0.3}
}

func (c *CondStageModel) setClipSkip(clipSkip int) {
	// Dummy implementation
}

func (c *CondStageModel) compute(nThreads int, inputIds, inputIds2 *ggmlTensor, maxTokenIdx int, flag bool, chunkHiddenStates **ggmlTensor, workCtx *ggmlContext) {
	// Dummy implementation
}

type DiffusionModel struct {
	unet struct {
		admInChannels int
	}
}

func ggmlTimeMs() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

func vectorToGgmlTensorI32(workCtx *ggmlContext, vec []int) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlDupTensor(workCtx *ggmlContext, tensor *ggmlTensor) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlTensorMean(tensor *ggmlTensor) float32 {
	// Dummy implementation
	return 0.0
}

func ggmlTensorGetF32(tensor *ggmlTensor, i0, i1, i2 int) float32 {
	// Dummy implementation
	return 0.0
}

func ggmlTensorSetF32(tensor *ggmlTensor, value float32, i0, i1, i2 int) {
	// Dummy implementation
}

func ggmlTensorScale(tensor *ggmlTensor, scale float32) {
	// Dummy implementation
}

func ggmlNelements(tensor *ggmlTensor) int {
	// Dummy implementation
	return 0
}

func vectorToGgmlTensor(workCtx *ggmlContext, vec []float32) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlReshape2D(workCtx *ggmlContext, tensor *ggmlTensor, dim0, dim1 int) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlNewTensor1D(workCtx *ggmlContext, tensorType int, size int) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlView2D(workCtx *ggmlContext, tensor *ggmlTensor, dim0, dim1, stride, offset int) *ggmlTensor {
	// Dummy implementation
	return &ggmlTensor{}
}

func ggmlNbytes(tensor *ggmlTensor) int {
	// Dummy implementation
	return 0
}

func setTimestepEmbedding(timesteps []float32, embedView *ggmlTensor, outDim int) {
	// Dummy implementation
}

func getLearnedCondition(workCtx *ggmlContext, text string, clipSkip, width, height int, forceZeroEmbeddings bool) (*ggmlTensor, *ggmlTensor) {
	tokens, weights := cond_stage_model.tokenize(text, true)
	return getLearnedConditionCommon(workCtx, tokens, weights, clipSkip, width, height, forceZeroEmbeddings)
}

func getLearnedConditionCommon(workCtx *ggmlContext, tokens []int, weights []float32, clipSkip, width, height int, forceZeroEmbeddings bool) (*ggmlTensor, *ggmlTensor) {
	cond_stage_model.setClipSkip(clipSkip)
	t0 := ggmlTimeMs()
	var hiddenStates, chunkHiddenStates, pooled *ggmlTensor
	var hiddenStatesVec []float32

	chunkLen := 77
	chunkCount := len(tokens) / chunkLen
	for chunkIdx := 0; chunkIdx < chunkCount; chunkIdx++ {
		chunkTokens := tokens[chunkIdx*chunkLen : (chunkIdx+1)*chunkLen]
		chunkWeights := weights[chunkIdx*chunkLen : (chunkIdx+1)*chunkLen]

		inputIds := vectorToGgmlTensorI32(workCtx, chunkTokens)
		var inputIds2 *ggmlTensor
		var maxTokenIdx int
		if version == VERSION_XL {
			it := bytes.IndexByte([]byte(fmt.Sprint(chunkTokens)), EOS_TOKEN_ID)
			if it != -1 {
				for i := it + 1; i < len(chunkTokens); i++ {
					chunkTokens[i] = 0
				}
			}

			maxTokenIdx = int(math.Min(float64(it), float64(len(chunkTokens)-1)))

			inputIds2 = vectorToGgmlTensorI32(workCtx, chunkTokens)
		}

		cond_stage_model.compute(n_threads, inputIds, inputIds2, maxTokenIdx, false, &chunkHiddenStates, workCtx)
		if version == VERSION_XL && chunkIdx == 0 {
			cond_stage_model.compute(n_threads, inputIds, inputIds2, maxTokenIdx, true, &pooled, workCtx)
		}

		t1 := ggmlTimeMs()
		fmt.Printf("computing condition graph completed, taking %d ms\n", t1-t0)
		result := ggmlDupTensor(workCtx, chunkHiddenStates)
		{
			originalMean := ggmlTensorMean(chunkHiddenStates)
			for i2 := 0; i2 < chunkHiddenStates.ne[2]; i2++ {
				for i1 := 0; i1 < chunkHiddenStates.ne[1]; i1++ {
					for i0 := 0; i0 < chunkHiddenStates.ne[0]; i0++ {
						value := ggmlTensorGetF32(chunkHiddenStates, i0, i1, i2)
						value *= chunkWeights[i1]
						ggmlTensorSetF32(result, value, i0, i1, i2)
					}
				}
			}
			newMean := ggmlTensorMean(result)
			ggmlTensorScale(result, originalMean/newMean)
		}
		if forceZeroEmbeddings {
			vec := result.data
			for i := 0; i < ggmlNelements(result); i++ {
				vec[i] = 0
			}
		}
		hiddenStatesVec = append(hiddenStatesVec, result.data...)
	}

	hiddenStates = vectorToGgmlTensor(workCtx, hiddenStatesVec)
	hiddenStates = ggmlReshape2D(workCtx, hiddenStates, chunkHiddenStates.ne[0], ggmlNelements(hiddenStates)/chunkHiddenStates.ne[0])

	var vec *ggmlTensor
	if version == VERSION_XL {
		outDim := 256
		vec = ggmlNewTensor1D(workCtx, 0, diffusion_model.unet.admInChannels)
		offset := 0
		copy(vec.data, pooled.data)
		offset += ggmlNbytes(pooled)

		origWidth := float32(width)
		origHeight := float32(height)
		timesteps := []float32{origHeight, origWidth}

		embedView := ggmlView2D(workCtx, vec, outDim, 2, 4*outDim, offset)
		offset += ggmlNbytes(embedView)
		setTimestepEmbedding(timesteps, embedView, outDim)

		cropCoordTop := float32(0)
		cropCoordLeft := float32(0)
		timesteps = []float32{cropCoordTop, cropCoordLeft}
		embedView = ggmlView2D(workCtx, vec, outDim, 2, 4*outDim, offset)
		offset += ggmlNbytes(embedView)
		setTimestepEmbedding(timesteps, embedView, outDim)

		targetWidth := float32(width)
		targetHeight := float32(height)
		timesteps = []float32{targetHeight, targetWidth}
		embedView = ggmlView2D(workCtx, vec, outDim, 2, 4*outDim, offset)
		offset += ggmlNbytes(embedView)
		setTimestepEmbedding(timesteps, embedView, outDim)
	}

	return hiddenStates, vec
}

func main() {
	// Example usage
	workCtx := &ggmlContext{}
	text := "example text"
	clipSkip := 1
	width := 256
	height := 256
	forceZeroEmbeddings := false

	hiddenStates, vec := getLearnedCondition(workCtx, text, clipSkip, width, height, forceZeroEmbeddings)
	fmt.Println(hiddenStates, vec)
}
