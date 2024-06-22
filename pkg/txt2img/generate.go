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

const (
	UNK_TOKEN_ID int = 49407
	BOS_TOKEN_ID int = 49406
	EOS_TOKEN_ID int = 49407
	PAD_TOKEN_ID int = 49407
)

// var encoder = make(map[string]int)

func (gen *Generator) Generate(prompt string, width, height int) (result *SDImage, err error) {

	// negativePrompt := ""
	clipSkip := 2
	// cfgScale := 1.0
	var sampleStepsInput = 16
	seed := uint64(42)

	var memSize uint64 = 10 * 1024 * 1024
	memSize += uint64(width * height * 3 * 4)

	var workCtx = gen.GGML.InitGo(memSize)
	if workCtx == nil {
		err = errors.New("gen.GGML.InitGo() failed")
		return
	}

	var schedule = KarrasSchedule{}

	sigmas := schedule.GetSigmas(sampleStepsInput)

	// Get learned condition
	// c, cVector := getLearnedCondition(workCtx, prompt, clipSkip, width, height, false)

	var pair = gen.GetLearnedCondition(gen.Model.GetCTX(), workCtx, prompt, width, height, clipSkip)
	var c = gen.PairGet(pair, true)
	var cVector = gen.PairGet(pair, false)

	// spew.Dump(c)
	// spew.Dump(cVector)

	// if sdCtx.SD.FreeParamsImmediately {
	// 	sdCtx.SD.CondStageModel.FreeParamsBuffer()
	// }

	// Sample
	C, W, H := 4, width/8, height/8

	// BATCH START
	// sdCtx.SD.RNG.Seed(seed)
	xT := gen.GGML.NewTensor4D(workCtx, 0, W, H, C, 1)

	gen.GGML.TensorSetF32Rand(xT, seed)

	// x0 := sdCtx.SD.SampleGo(workCtx, xT, c, cVector, sigmas)

	var x0 = gen.GoSample(gen.Model.GetCTX(), workCtx, xT, c, cVector, len(sigmas), sigmas)

	// BATCH END

	spew.Dump(x0)

	/*

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
