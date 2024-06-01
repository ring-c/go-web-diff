package generate

import (
	"encoding/json"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

type InputData struct {
	Options opts.Options `json:"options"`
	Params  opts.Params  `json:"params"`
}

var DefaultInput = &InputData{
	Options: opts.Options{
		Threads:               -1,
		VaeDecodeOnly:         true,
		VaeTiling:             true,
		FreeParamsImmediately: false,
		RngType:               opts.STD_DEFAULT_RNG,
		WType:                 opts.F16,
		Schedule:              opts.KARRAS,
		GpuEnable:             true,
		Debug:                 true,
	},
	Params: opts.Params{
		WithUpscale:      false,
		DeleteUpscaled:   false,
		OutputDir:        "./output/",
		Prompt:           "1girl",
		NegativePrompt:   "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
		CfgScale:         7.0,
		ClipSkip:         2,
		Width:            512,
		Height:           512,
		SampleMethod:     opts.EULER_A,
		SampleSteps:      24,
		Strength:         0.7,
		Seed:             42,
		BatchCount:       1,
		OutputsImageType: opts.PNG,
	},
}

func getInput(c echo.Context) (data *InputData, err error) {
	data = DefaultInput

	// Overwrite from request
	err = c.Bind(data)
	if err != nil {
		return
	}

	return
}

func GetInputFromJSON(input []byte) (data *InputData, err error) {
	data = DefaultInput

	// Overwrite from request
	err = json.Unmarshal(input, data)
	if err != nil {
		return
	}

	return
}
