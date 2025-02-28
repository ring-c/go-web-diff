package handlers

import (
	"errors"
	"strings"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

var DefaultInput = &opts.Options{
	RngType:   opts.CUDA_RNG,
	WType:     opts.GGML_TYPE_COUNT,
	Schedule:  opts.DEFAULT,
	GpuEnable: true,
	Debug:     true,

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

	if data.Width%8 != 0 || data.Height%8 != 0 {
		err = errors.New("width and height must be multiples of 8")
		return
	}

	if data.UseModelSD && data.UseModelFlux {
		err = errors.New("UseModelSD AND UseModelFlux at the same time not allowed")
		return
	}

	if data.UseModelFlux {
		data.ModelPath = ""
	}

	if data.UseModelSD {
		data.FluxModelPath = ""
		data.ClipLPath = ""
		data.T5xxlPath = ""
	}

	data.Prompt = parsePrompt(data.Prompt, true)
	data.NegativePrompt = parsePrompt(data.NegativePrompt, true)
	data.Lora = parsePrompt(data.Lora, false)
	return
}

func parsePrompt(input string, withGroups bool) string {
	var data = make([]string, 0)
	for _, str := range strings.Split(input, "\n") {
		str = strings.TrimSpace(str)
		if len(str) < 1 {
			continue
		}

		if strings.HasPrefix(str, "#") || strings.HasPrefix(str, "//") {
			continue
		}

		// var text = fmt.Sprintf("%s", str)
		// if !withGroups || (strings.Contains(str, "(") && strings.Contains(str, ":") && strings.Contains(str, ")")) {
		// 	text = str
		// }

		data = append(data, str)
	}

	return strings.Join(data, ", ")
}
