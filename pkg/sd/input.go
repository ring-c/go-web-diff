package sd

import (
	"encoding/json"
	"os"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type inputFile struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"prompt_neg"`
	Seed           int64  `json:"seed"`
}

func GetInput() (params *opts.FullParams, err error) {
	params = DefaultFullParams

	data, err := os.ReadFile("input.json")
	if err != nil {
		return
	}

	var in inputFile
	err = json.Unmarshal(data, &in)
	if err != nil {
		return
	}

	params.Prompt = in.Prompt
	params.NegativePrompt = in.NegativePrompt
	params.Seed = in.Seed

	params.Width = 544
	params.Height = 960
	params.ClipSkip = 2
	params.CfgScale = 7
	params.SampleSteps = 22
	params.SampleMethod = opts.EULER_A
	params.BatchCount = 9

	return
}
