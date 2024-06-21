package txt2img

import (
	"errors"
	"math/rand"

	"github.com/davecgh/go-spew/spew"
)

type SDContext struct {
	SD *SDModel
}

type SDModel struct {
	Denoiser              *Denoiser
	CondStageModel        *CondStageModel
	DiffusionModel        *DiffusionModel
	FirstStageModel       *FirstStageModel
	RNG                   *rand.Rand
	FreeParamsImmediately bool
	UseTinyAutoencoder    bool
}

type Denoiser struct {
	Schedule *Schedule
}

type Schedule struct {
	Sigmas []float32
}

type CondStageModel struct {
	ParamsBuffer []float32
}

type DiffusionModel struct {
	schedule     *Schedule
	ParamsBuffer []float32
	uNet         struct {
		admInChannels int
	}
}

type FirstStageModel struct {
	ParamsBuffer []float32
}

func (gen *Generator) Generate(sdCtx *SDContext, prompt string, width, height int) (result *SDImage, err error) {
	// negativePrompt := ""
	// clipSkip := 2
	// cfgScale := 1.0
	var sampleStepsInput uint32 = 16
	// seed := int64(42)

	var memSize uint64 = 10 * 1024 * 1024
	memSize += uint64(width * height * 3 * 4)

	workCtx := gen.GGML.InitGo(memSize)
	if workCtx == nil {
		err = errors.New("gen.GGML.InitGo() failed")
		return
	}

	var schedule = KarrasSchedule{}

	sigmas := schedule.GetSigmas(sampleStepsInput)

	spew.Dump(sigmas)

	/*

		// Get learned condition
		c, cVector := getLearnedCondition(workCtx, prompt, clipSkip, width, height, false)

		spew.Dump(c)
		spew.Dump(cVector)



			if sdCtx.SD.FreeParamsImmediately {
				sdCtx.SD.CondStageModel.FreeParamsBuffer()
			}

			// Sample
			C, W, H := 4, width/8, height/8

			// BATCH START
			sdCtx.SD.RNG.Seed(seed)
			xT := GGMLNewTensor4D(workCtx, GGMLTypeF32, W, H, C, 1)
			GGMLTensorSetF32Randn(xT, sdCtx.SD.RNG)

			x0 := sdCtx.SD.SampleGo(workCtx, xT, c, cVector, sigmas)

			// BATCH END

			if sdCtx.SD.FreeParamsImmediately {
				sdCtx.SD.DiffusionModel.FreeParamsBuffer()
			}

			// Decode to image
			decodedImages := make([]*GGMLTensor, 0)
			img := sdCtx.SD.DecodeFirstStage(workCtx, x0)
			if img != nil {
				decodedImages = append(decodedImages, img)
			}

			log.Info("decode_first_stage completed")
			if sdCtx.SD.FreeParamsImmediately && !sdCtx.SD.UseTinyAutoencoder {
				sdCtx.SD.FirstStageModel.FreeParamsBuffer()
			}

			result := &SDImage{
				Width:   width,
				Height:  height,
				Channel: 3,
				Data:    SDTensorToImage(decodedImages[0]),
			}
			// GGMLFree(workCtx)
	*/
	return
}

type SDImage struct {
	Width   int
	Height  int
	Channel int
	Data    []byte
}

type GGMLInitParams struct {
	MemSize   int64
	MemBuffer []byte
	NoAlloc   bool
}

func (s *CondStageModel) FreeParamsBuffer() {
	// Implement FreeParamsBuffer method
}

func (s *DiffusionModel) FreeParamsBuffer() {
	// Implement FreeParamsBuffer method
}

func (s *FirstStageModel) FreeParamsBuffer() {
	// Implement FreeParamsBuffer method
}

type GGMLContext struct {
}

type GGMLTensor struct {
	data []float32
	ne   [4]int
}

type GGMLType int

type ConditionPair struct {
	First  *GGMLTensor
	Second *GGMLTensor
}
