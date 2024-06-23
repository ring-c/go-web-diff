package txt2img

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
	"os"
	"path"
	"time"
	"unsafe"
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

func (gen *Generator) Generate(prompt string, width, height int) (err error) {

	// negativePrompt := ""
	// clipSkip := 2
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

	var sigmas = schedule.GetSigmas(sampleStepsInput)
	// spew.Dump(sigmas)

	// Get learned condition
	// c, cVector := getLearnedCondition(workCtx, prompt, clipSkip, width, height, false)

	// var pairCond = gen.GetLearnedCondition(gen.Model.GetCTX(), workCtx, prompt, width, height, clipSkip)

	// spew.Dump(pairCond)

	// var c = gen.PairGet(pair, true)
	// var cVector = gen.PairGet(pair, false)

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

	var cImageData = gen.GoSample(gen.Model.GetCTX(), workCtx, xT, len(sigmas), sigmas)

	// BATCH END

	var imgData = unsafe.Slice((*byte)(cImageData), 3*width*height)
	var data = bytesToImage(imgData, width, height)

	var filename = fmt.Sprintf("%d-%d.png", time.Now().Unix(), seed)
	var file *os.File
	file, err = os.Create(path.Join(".", filename))
	if err != nil {
		return
	}

	err = imageToWriter(data, file)
	if err != nil {
		return
	}

	err = file.Close()
	if err != nil {
		return
	}

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

func bytesToImage(byteData []byte, width, height int) (img *image.RGBA) {
	img = image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := (y*width + x) * 3
			img.Set(x, y, color.RGBA{
				R: byteData[idx],
				G: byteData[idx+1],
				B: byteData[idx+2],
				A: 255,
			})
		}
	}

	return
}

func imageToWriter(image *image.RGBA, writer io.Writer) (err error) {
	var enc = png.Encoder{
		CompressionLevel: png.BestSpeed,
	}

	err = enc.Encode(writer, image)
	if err != nil {
		return
	}

	return
}
