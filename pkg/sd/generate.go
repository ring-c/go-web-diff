package sd

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

func (sd *Model) Generate(in *opts.Options) (filenames []string, err error) {
	// var timeTotalStart = time.Now()

	if sd.GetCTX() == nil {
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

	sd.filenames = make([]string, 0)

	var seed = in.Seed
	if seed == -1 {
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

		prompt += fmt.Sprintf("<lora:%s>", lora)
	}

	var data = sd.cSD.Text2Image(
		sd.GetCTX(),
		prompt, in.NegativePrompt,
		in.ClipSkip,
		in.CfgScale, 3.50, in.Width, in.Height,
		int(in.SampleMethod), in.SampleSteps,
		seed, in.BatchCount, nil, 0.9, 20, false, "",
	)

	for _, img := range goImageSlice(data, in.BatchCount) {
		var imgData = &img
		sd.fileWrite.Add(1)
		go sd.writeFile(imgData, in, seed)
	}

	sd.fileWrite.Wait()

	filenames = sd.filenames
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

func (sd *Model) writeFile(img *Image, in *opts.Options, seed int64) {
	defer func() {
		sd.fileWrite.Done()
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

	sd.filenames = append(sd.filenames, filename)
}
