package sd

import (
	"fmt"
	"math/rand"
	"time"
)

func generateImage(
	sdCtx *sd_ctx_t, workCtx *ggml_context, initLatent *ggml_tensor, prompt string, negativePrompt string, clipSkip int, cfgScale float32,
	width int, height int, sampleMethod sample_method_t, sigmas []float64, seed int64, batchCount int, controlCond *sd_image_t,
	controlStrength float32, styleRatio float32, normalizeInput bool, inputIdImagesPath string,
) *sd_image_t {
	if seed < 0 {
		rand.Seed(time.Now().Unix())
		seed = int64(rand.Int())
	}

	sampleSteps := len(sigmas) - 1

	resultPair := extractAndRemoveLora(prompt)
	loraF2m := resultPair.first

	for key, value := range loraF2m {
		fmt.Printf("lora %s:%.2f\n", key, value)
	}

	prompt = resultPair.second
	fmt.Printf("prompt after extract and remove lora: \"%s\"\n", prompt)

	t0 := ggmlTimeMs()
	sdCtx.sd.applyLoras(loraF2m)
	t1 := ggmlTimeMs()
	fmt.Printf("apply_loras completed, taking %.2fs\n", float32(t1-t0)/1000)

	// var promptTextOnly string
	// var initImg *ggml_tensor
	var promptsEmbeds *ggml_tensor
	var pooledPromptsEmbeds *ggml_tensor
	// var classTokensMask []bool

	/*
		if sdCtx.sd.stackedId {
			if !sdCtx.sd.pmidLora.applied {
				t0 = ggmlTimeMs()
				sdCtx.sd.pmidLora.apply(sdCtx.sd.tensors, sdCtx.sd.nThreads)
				t1 = ggmlTimeMs()
				sdCtx.sd.pmidLora.applied = true
				fmt.Printf("pmid_lora apply completed, taking %.2fs\n", float32(t1-t0)/1000)
				if sdCtx.sd.freeParamsImmediately {
					sdCtx.sd.pmidLora.freeParamsBuffer()
				}
			}

			var inputIdImages []*sd_image_t
			if sdCtx.sd.pmidModel != nil && len(inputIdImagesPath) > 0 {
				imgFiles := getFilesFromDir(inputIdImagesPath)
				for _, imgFile := range imgFiles {
					c := 0
					var width, height int
					inputImageBuffer := stbiLoad(imgFile, &width, &height, &c, 3)
					if inputImageBuffer == nil {
						fmt.Printf("PhotoMaker load image from '%s' failed\n", imgFile)
						continue
					} else {
						fmt.Printf("PhotoMaker loaded image from '%s'\n", imgFile)
					}
					inputImage := &sd_image_t{
						width:   uint32(width),
						height:  uint32(height),
						channel: 3,
						data:    inputImageBuffer,
					}
					inputImage = preprocessIdImage(inputImage)
					if inputImage == nil {
						fmt.Printf("preprocess input id image from '%s' failed\n", imgFile)
						continue
					}
					inputIdImages = append(inputIdImages, inputImage)
				}
			}
			if len(inputIdImages) > 0 {
				sdCtx.sd.pmidModel.styleStrength = styleRatio
				w := inputIdImages[0].width
				h := inputIdImages[0].height
				channels := inputIdImages[0].channel
				numInputImages := int32(len(inputIdImages))
				initImg = ggmlNewTensor4d(workCtx, GGML_TYPE_F32, w, h, channels, numInputImages)
				mean := []float32{0.48145466, 0.4578275, 0.40821073}
				std := []float32{0.26862954, 0.26130258, 0.27577711}
				for i := 0; i < len(inputIdImages); i++ {
					initImage := inputIdImages[i]
					if normalizeInput {
						sdMulImagesToTensor(initImage.data, initImg, int32(i), mean, std)
					} else {
						sdMulImagesToTensor(initImage.data, initImg, int32(i), nil, nil)
					}
				}
				t0 = ggmlTimeMs()
				condTup := sdCtx.sd.getLearnedConditionWithTrigger(workCtx, prompt, clipSkip, width, height, numInputImages)
				promptsEmbeds = condTup.first
				pooledPromptsEmbeds = condTup.second
				classTokensMask = condTup.third

				promptsEmbeds = sdCtx.sd.idEncoder(workCtx, initImg, promptsEmbeds, classTokensMask)
				t1 = ggmlTimeMs()
				fmt.Printf("Photomaker ID Stacking, taking %" PRId64 " ms\n", t1-t0)
				if sdCtx.sd.freeParamsImmediately {
					sdCtx.sd.pmidModel.freeParamsBuffer()
				}
				promptTextOnly = sdCtx.sd.removeTriggerFromPrompt(workCtx, prompt)
				prompt = promptTextOnly
			} else {
				fmt.Println("Provided PhotoMaker model file, but NO input ID images")
				fmt.Println("Turn off PhotoMaker")
				sdCtx.sd.stackedId = false
			}
			for _, img := range inputIdImages {
				free(img.data)
			}
			inputIdImages = nil
		}
	*/

	t0 = ggmlTimeMs()
	condPair := sdCtx.sd.getLearnedCondition(workCtx, prompt, clipSkip, width, height)
	c := condPair.first
	cVector := condPair.second

	var uc *ggml_tensor
	var ucVector *ggml_tensor
	if cfgScale != 1.0 {
		forceZeroEmbeddings := false
		if sdCtx.sd.version == VERSION_XL && len(negativePrompt) == 0 {
			forceZeroEmbeddings = true
		}
		uncondPair := sdCtx.sd.getLearnedCondition(workCtx, negativePrompt, clipSkip, width, height, forceZeroEmbeddings)
		uc = uncondPair.first
		ucVector = uncondPair.second
	}
	t1 = ggmlTimeMs()
	// fmt.Printf("get_learned_condition completed, taking %" PRId64 " ms\n", t1-t0)

	if sdCtx.sd.freeParamsImmediately {
		sdCtx.sd.condStageModel.freeParamsBuffer()
	}

	var imageHint *ggml_tensor
	if controlCond != nil {
		imageHint = ggmlNewTensor4d(workCtx, GGML_TYPE_F32, width, height, 3, 1)
		sdImageToTensor(controlCond.data, imageHint)
	}

	var finalLatents []*ggml_tensor
	C := 4
	W := width / 8
	H := height / 8
	fmt.Printf("sampling using %s method\n", samplingMethodsStr[sampleMethod])
	for b := 0; b < batchCount; b++ {
		samplingStart := ggmlTimeMs()
		curSeed := seed + int64(b)
		fmt.Printf("generating image: %d/%d - seed %d\n", b+1, batchCount, curSeed)

		sdCtx.sd.rng.manualSeed(curSeed)
		var xT *ggml_tensor
		var noise *ggml_tensor
		if initLatent == nil {
			xT = ggmlNewTensor4d(workCtx, GGML_TYPE_F32, W, H, C, 1)
			ggmlTensorSetF32Randn(xT, sdCtx.sd.rng)
		} else {
			xT = initLatent
			noise = ggmlNewTensor4d(workCtx, GGML_TYPE_F32, W, H, C, 1)
			ggmlTensorSetF32Randn(noise, sdCtx.sd.rng)
		}

		startMergeStep := -1
		if sdCtx.sd.stackedId {
			startMergeStep = int(sdCtx.sd.pmidModel.styleStrength / 100.0 * float32(sampleSteps))
			fmt.Printf("PHOTOMAKER: start_merge_step: %d\n", startMergeStep)
		}

		x0 := sample(workCtx, xT, noise, c, nil, cVector, uc, nil, ucVector, imageHint, controlStrength, cfgScale, cfgScale, sampleMethod, sigmas, startMergeStep, promptsEmbeds, pooledPromptsEmbeds)
		samplingEnd := ggmlTimeMs()
		fmt.Printf("sampling completed, taking %.2fs\n", float32(samplingEnd-samplingStart)/1000)
		finalLatents = append(finalLatents, x0)
	}

	// TODO: return the resulting sd_image_t
	return nil
}
