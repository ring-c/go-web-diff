package txt2img

import (
	"unsafe"
)

type CondStageModel struct {
	ParamsBuffer []float32
}

func (c *CondStageModel) setClipSkip(clipSkip int) {
	panic("fix me")
}

func (c *CondStageModel) compute(nThreads int, inputIds, inputIds2 *ggmlTensor, maxTokenIdx int, flag bool, chunkHiddenStates **ggmlTensor, workCtx unsafe.Pointer) {
	panic("fix me")
}

func buildGraph(inputIDs, inputIDs2 *ggml.Tensor, maxTokenIdx uint64, returnPooled bool) *ggml.Graph {
	gf := ggml.NewGraph(computeCtx)

	if inputIDs2 != nil {
		inputIDs2 = toBackend(inputIDs2)
	}
	if !returnPooled {
		inputIDs = toBackend(inputIDs)
	}

	var embeddings *ggml.Tensor

	if numCustomEmbeddings > 0 && version != versionXL {
		customEmbeddings := ggml.NewTensor3D(computeCtx, wtype, textModel.HiddenSize, 1, numCustomEmbeddings)
		setBackendTensorData(customEmbeddings, tokenEmbedCustom.Data())

		tokenEmbedWeight := textModel.GetTokenEmbedWeight()
		tokenEmbedWeight = ggml.Reshape3D(computeCtx, tokenEmbedWeight, tokenEmbedWeight.Ne[0], 1, tokenEmbedWeight.Ne[1])
		embeddings = ggml.Concat(computeCtx, tokenEmbedWeight, customEmbeddings)
		embeddings = ggml.Reshape2D(computeCtx, embeddings, embeddings.Ne[0], embeddings.Ne[2])
	}

	hiddenStates := forward(computeCtx, inputIDs, inputIDs2, embeddings, maxTokenIdx, returnPooled)

	ggml.BuildForwardExpand(gf, hiddenStates)

	return gf
}

func compute(nThreads int, inputIDs, inputIDs2 *ggml.Tensor, maxTokenIdx uint64, returnPooled bool, output **ggml.Tensor, outputCtx *ggml.Context) {
	getGraph := func() *ggml.Graph {
		return buildGraph(inputIDs, inputIDs2, maxTokenIdx, returnPooled)
	}
	GGMLModule.Compute(getGraph, nThreads, true, output, outputCtx)
}
