package generate

import (
	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
)

type inputFile struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"prompt_neg"`
	Seed           int64  `json:"seed"`
}

func getInput(c echo.Context) (params *opts.FullParams, err error) {
	params = sd.DefaultFullParams

	var data inputFile

	err = c.Bind(&data)
	if err != nil {
		return
	}

	params.Prompt = data.Prompt
	params.NegativePrompt = data.NegativePrompt
	params.Seed = data.Seed

	params.Width = 544
	params.Height = 960
	params.ClipSkip = 2
	params.CfgScale = 7
	params.SampleSteps = 32
	params.SampleMethod = opts.EULER_A
	params.BatchCount = 4

	return
}
