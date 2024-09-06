package generate

import (
	"strings"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

var DefaultInput = &opts.Options{
	RngType:   opts.CUDA_RNG,
	WType:     opts.F16,
	Schedule:  opts.DEFAULT,
	VaeTiling: true,
	GpuEnable: true,
	Debug:     true,
	Threads:   10,

	OutputDir: "./output/",

	CfgScale:         7.0,
	ClipSkip:         2,
	Width:            512,
	Height:           512,
	SampleMethod:     opts.EULER_A,
	SampleSteps:      24,
	Seed:             42,
	BatchCount:       1,
	OutputsImageType: opts.PNG,
}

func getInput(c echo.Context) (data *opts.Options, err error) {
	data = DefaultInput

	// Overwrite from request
	err = c.Bind(data)
	if err != nil {
		return
	}

	data.Prompt = parsePrompt(data.Prompt)
	data.NegativePrompt = parsePrompt(data.NegativePrompt)
	data.Loras = parsePrompt(data.Loras)
	return
}

func parsePrompt(input string) string {
	var data = make([]string, 0)
	for _, str := range strings.Split(input, "\n") {
		if len(str) < 1 {
			continue
		}

		if strings.HasPrefix(str, "#") || strings.HasPrefix(str, "//") {
			continue
		}

		data = append(data, str)
	}

	return strings.Join(data, ", ")
}
