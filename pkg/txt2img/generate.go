package txt2img

import (
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"path"
	"time"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (gen *Generator) Generate(in *opts.Options) (filenames []string, err error) {
	filenames = make([]string, 0)
	if gen.Model.GetCTX() == nil {
		err = errors.New("model not loaded")
		return
	}

	if in == nil {
		err = errors.New("options is nil")
		return
	}

	if in.Width%8 != 0 || in.Height%8 != 0 {
		err = errors.New("width and height must be multiples of 8")
		return
	}

	var seed = in.Seed
	if seed == 0 {
		seed = rand.Uint64()
		if in.Debug {
			fmt.Printf("Generating random starting seed: %d\n", seed)
		}
	}

	// negativePrompt := ""
	// clipSkip := 2
	// cfgScale := 1.0

	var memSize uint64 = 10 * 1024 * 1024
	memSize += uint64(in.Width * in.Height * 3 * 4)

	var schedule = KarrasSchedule{}

	var sigmas = schedule.GetSigmas(in.SampleSteps)
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
	var W = in.Width / 8
	var H = in.Height / 8

	var timeSave = time.Now().Unix()
	for i := 0; i < in.BatchCount; i++ {
		if in.Debug {
			fmt.Printf("[%d/%d] Generating with seed %d\n", i+1, in.BatchCount, seed)
		}

		var timeStart = time.Now()

		var workCtx = gen.GGML.Init(memSize)
		if workCtx == nil {
			err = errors.New("gen.GGML.Init() failed")
			return
		}

		xT := gen.GGML.NewTensor4D(workCtx, 0, W, H, 4, 1)

		gen.GGML.TensorSetF32Rand(xT, seed)

		if in.Debug {
			fmt.Printf("[%d/%d] Prep done in %gs\n", i+1, in.BatchCount, time.Now().Sub(timeStart).Seconds())
		}

		var cImageDataPointer = gen.GoSample(gen.Model.GetCTX(), workCtx, xT, in.Prompt, len(sigmas), sigmas)
		var decoded = gen.computeFirstStage(workCtx, cImageDataPointer, in)

		if in.Debug {
			fmt.Printf("[%d/%d] Done in %gs\n", i+1, in.BatchCount, time.Now().Sub(timeStart).Seconds())
		}

		var data = gen.sdTensorToImage(decoded, in)

		var filename = fmt.Sprintf("%d-%d.png", timeSave, seed)
		var file *os.File
		file, err = os.Create(path.Join("./output/", filename))
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

		gen.GGML.Free(workCtx)

		filenames = append(filenames, filename)
		seed++
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
