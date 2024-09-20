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

	sd.fileWrite.Add(in.BatchCount)
	sd.cSD.SDSetResultCallback(sd.ResultCallback, in)

	_ = sd.cSD.Text2Image(
		sd.GetCTX(),
		prompt, in.NegativePrompt,
		in.ClipSkip,
		in.CfgScale, 3.50, in.Width, in.Height,
		int(in.SampleMethod), in.SampleSteps,
		seed, in.BatchCount, nil, 0.9, 20, false, "",
	)

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

func (sd *Model) writeFile(data []byte, in *opts.Options, num uint64, seed int64) {
	defer func() {
		sd.fileWrite.Done()
	}()

	var filename = fmt.Sprintf("%d-%d.png", time.Now().Unix(), seed+int64(num-1))
	var file *os.File
	file, err := os.Create(path.Join(in.OutputDir, filename))
	if err != nil {
		println("writeFile os.Create:" + err.Error())
		return
	}

	var outputImg = bytesToImage(data, in.Width, in.Height)
	err = imageToWriter(outputImg, file)
	if err != nil {
		println("writeFile imageToWriter:" + err.Error())
		return
	}

	err = file.Close()
	if err != nil {
		println("writeFile file.Close:" + err.Error())
		return
	}

	sd.filenames[num-1] = filename
}
