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
	if sd.GetCTX() == nil {
		err = errors.New("model not loaded")
		return
	}

	if in == nil {
		err = errors.New("options is nil")
		return
	}

	sd.filenames = make([]string, in.BatchCount)

	var seed = in.Seed
	if seed == -1 {
		seed = rand.Int63()
	}

	var prompt = in.Prompt
	for _, lora := range strings.Split(in.Lora, ", ") {
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

	sd.fileWrite.Add(in.BatchCount)
	for num, img := range goImageSlice(data, in.BatchCount) {
		var imgData = &img
		var num = num

		go sd.writeFile(imgData, in, num, seed)
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

func (sd *Model) writeFile(img *Image, in *opts.Options, num int, seed int64) {
	defer func() {
		sd.fileWrite.Done()
	}()

	var outputImg = bytesToImage(img.Data, in.Width, in.Height)

	var filename = fmt.Sprintf("%d-%d-%d.png", time.Now().Unix(), num, seed)
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

	sd.filenames[num] = filename
}
