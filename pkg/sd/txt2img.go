package sd

import (
	"errors"
	"fmt"
	"strings"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (sd *Model) Txt2Img(in *opts.Options) (filenames []string, err error) {
	if sd.GetCTX() == nil {
		err = errors.New("model not loaded")
		return
	}

	if in == nil {
		err = errors.New("options is nil")
		return
	}

	var prompt = in.Prompt
	for _, lora := range strings.Split(in.Lora, ", ") {
		lora = strings.TrimSpace(lora)
		if lora == "" {
			continue
		}

		prompt += fmt.Sprintf("<lora:%s>", lora)
	}

	var generation = opts.NewGeneration(in)

	sd.cSD.SDSetResultCallback(sd.GetCTX(), resultCallback, generation)
	// sd.cSD.SDSetResultStepCallback(sd.GetCTX(), resultStepCallback, generation)

	sd.cSD.Text2Image(
		sd.GetCTX(),
		prompt, in.NegativePrompt,
		in.ClipSkip, in.CfgScale, 3.5,
		in.Width, in.Height,
		int(in.SampleMethod), in.SampleSteps,
		generation.Seed, in.BatchCount, nil, 0.9, 20, false, "",
	)

	filenames = generation.GetFilenames()
	return
}
