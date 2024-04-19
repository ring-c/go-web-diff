package generate

import (
	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func getInput(c echo.Context) (params *opts.Params, err error) {
	params = &opts.Params{
		Prompt:           "1girl",
		NegativePrompt:   "out of frame, lowers, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
		CfgScale:         7.0,
		ClipSkip:         2,
		Width:            512,
		Height:           512,
		SampleMethod:     opts.EULER_A,
		SampleSteps:      24,
		Strength:         0.4,
		Seed:             42,
		BatchCount:       1,
		OutputsImageType: opts.PNG,
	}

	err = c.Bind(params)
	if err != nil {
		return
	}

	return
}
