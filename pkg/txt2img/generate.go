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
	"strings"
	"time"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (gen *Generator) Generate(in *opts.Options) (filenames []string, err error) {
	var timeTotalStart = time.Now()

	gen.filenames = make([]string, 0)

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
		seed = rand.Int63()
		if in.Debug {
			fmt.Printf("Generating random starting seed: %d\n", seed)
		}
	}

	var prompt = in.Prompt
	for _, lora := range strings.Split(in.Loras, ", ") {
		lora = strings.TrimSpace(lora)
		if lora == "" {
			continue
		}

		prompt += fmt.Sprintf("<lora:%s>, ", lora)
	}

	for i := 0; i < in.BatchCount; i++ {
		if in.Debug {
			fmt.Printf("[%d/%d] Generating with seed %d\n", i+1, in.BatchCount, seed)
		}

		var timeStart = time.Now()

		var data = gen.Text2Image(
			gen.Model.GetCTX(),
			prompt, in.NegativePrompt,
			2, in.CfgScale, 0, in.Width, in.Height, int(in.SampleMethod),
			in.SampleSteps, seed, in.BatchCount, nil, 0, 0, false, "",
		)

		var images = goImageSlice(data, 1)

		gen.fileWrite.Add(1)
		go gen.writeFile(&images[0], in, seed)

		if in.Debug {
			fmt.Printf("[%d/%d] Done in %gs\n", i+1, in.BatchCount, time.Now().Sub(timeStart).Seconds())
		}

		seed++
	}

	if in.Debug {
		fmt.Printf("Total Done in %gs\n", time.Now().Sub(timeTotalStart).Seconds())
	}

	gen.fileWrite.Wait()

	filenames = gen.filenames
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

func (gen *Generator) writeFile(img *Image, in *opts.Options, seed int64) {
	defer func() {
		gen.fileWrite.Done()
	}()

	var outputImg = bytesToImage(img.Data, in.Width, in.Height)

	var filename = fmt.Sprintf("%d-%d.png", time.Now().Unix(), seed)
	var file *os.File
	file, err := os.Create(path.Join(in.OutputDir, filename))
	if err != nil {
		println("writeFile:" + err.Error())
		return
	}

	err = imageToWriter(outputImg, file)
	if err != nil {
		println("writeFile:" + err.Error())
		return
	}

	err = file.Close()
	if err != nil {
		println("writeFile:" + err.Error())
		return
	}

	gen.filenames = append(gen.filenames, filename)
}
