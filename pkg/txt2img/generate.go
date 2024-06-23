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

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (gen *Generator) Generate(in *opts.Options) (filenames []string, err error) {
	// negativePrompt := ""
	// clipSkip := 2
	// cfgScale := 1.0
	var sampleStepsInput = 16
	seed := uint64(rand.Int63())

	var memSize uint64 = 10 * 1024 * 1024
	memSize += uint64(in.Width * in.Height * 3 * 4)

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
	C, W, H := 4, in.Width/8, in.Height/8

	// BATCH START
	xT := gen.GGML.NewTensor4D(workCtx, 0, W, H, C, 1)
	gen.GGML.TensorSetF32Rand(xT, seed)

	var cImageData = gen.GoSample(gen.Model.GetCTX(), workCtx, xT, in.Prompt, len(sigmas), sigmas)

	// BATCH END

	var imgData = unsafe.Slice((*byte)(cImageData), 3*in.Width*in.Height)
	var data = bytesToImage(imgData, in.Width, in.Height)

	var filename = fmt.Sprintf("%d-%d.png", time.Now().Unix(), seed)
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

	filenames = make([]string, 0)
	filenames = append(filenames, filename)

	return
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
