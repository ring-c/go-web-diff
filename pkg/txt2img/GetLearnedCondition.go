//go:build tests

package txt2img

import (
	"math"
	"unsafe"
)

type ggmlTensor struct {
	data []float32
	ne   [3]int
}

var (
	nThreads = 4
)

func setTimestepEmbedding(timesteps []float32, embedView *ggmlTensor, outDim int) {
	panic("fix me")
}

func (gen *Generator) GetLearnedCondition(workCtx unsafe.Pointer, prompt string, clipSkip, width, height int, forceZeroEmbeddings bool) (unsafe.Pointer, unsafe.Pointer) {
	tokens, weights := gen.Tokenize(prompt, true)
	return gen.getLearnedConditionCommon(workCtx, tokens, weights, clipSkip, width, height, forceZeroEmbeddings)
}

func (gen *Generator) getLearnedConditionCommon(workCtx unsafe.Pointer, tokens []int, weights []float32, clipSkip, width, height int, forceZeroEmbeddings bool) (unsafe.Pointer, unsafe.Pointer) {
	cond_stage_model.setClipSkip(clipSkip)
	var hiddenStates, chunkHiddenStates, pooled *ggmlTensor
	var hiddenStatesVec []float32

	chunkLen := 77
	chunkCount := len(tokens) / chunkLen
	for chunkIdx := 0; chunkIdx < chunkCount; chunkIdx++ {
		chunkTokens := tokens[chunkIdx*chunkLen : (chunkIdx+1)*chunkLen]
		chunkWeights := weights[chunkIdx*chunkLen : (chunkIdx+1)*chunkLen]

		inputIds := gen.GGML.VectorToGgmlTensorI32(workCtx, chunkTokens)
		var inputIds2 *ggmlTensor
		var maxTokenIdx int
		// if version == VERSION_XL {

		// FIXME?
		var it = 0
		for i, token := range chunkTokens {
			if token == EOS_TOKEN_ID {
				chunkTokens[i] = 0
				it = i
			}
		}

		maxTokenIdx = int(math.Min(float64(it), float64(len(chunkTokens)-1)))

		inputIds2 = vectorToGgmlTensorI32(workCtx, chunkTokens)
		// }

		cond_stage_model.compute(nThreads, inputIds, inputIds2, maxTokenIdx, false, &chunkHiddenStates, workCtx)
		if chunkIdx == 0 {
			cond_stage_model.compute(nThreads, inputIds, inputIds2, maxTokenIdx, true, &pooled, workCtx)
		}

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

	outDim := 256
	vec = ggmlNewTensor1D(workCtx, 0, diffusion_model.uNet.admInChannels)
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

	return hiddenStates, vec
}
