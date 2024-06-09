package generate

import (
	"strings"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

var DefaultInput = &opts.Options{
	Threads:               -1,
	VaeDecodeOnly:         false,
	VaeTiling:             true,
	FreeParamsImmediately: false,
	RngType:               opts.STD_DEFAULT_RNG,
	WType:                 opts.F16,
	Schedule:              opts.KARRAS,
	GpuEnable:             true,
	Debug:                 true,

	WithUpscale:      false,
	DeleteUpscaled:   false,
	OutputDir:        "./output/",
	Prompt:           "",
	NegativePrompt:   "",
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
	return
}

func parsePrompt(input string) (prompt string) {
	var LORAs = make([]string, 0)
	for _, str := range strings.Split(input, "\n") {
		if len(str) < 1 {
			continue
		}

		if strings.HasPrefix(str, "#") || strings.HasPrefix(str, "//") {
			continue
		}

		if strings.HasPrefix(str, "<lora") {
			LORAs = append(LORAs, str)
			continue
		}

		prompt += str + ", "
	}

	prompt += strings.Join(LORAs, "")
	return
}
