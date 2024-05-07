package generate

import (
	"fmt"
	"net/http"
	"path/filepath"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/sd"
)

func Generate(c echo.Context) (err error) {
	in, err := getInput(c)
	if err != nil {
		println(err.Error())
		return
	}

	// spew.Dump(in)

	err = Run(in)
	if err != nil {
		println(err.Error())
		return
	}

	fmt.Printf("\nDONE\n")

	_ = c.JSON(http.StatusOK, "OK")
	return
}

func Run(in *InputData) (err error) {
	model, err := sd.NewModel(&in.Options)
	if err != nil {
		return
	}
	defer model.Close()

	// println(model.GetSystemInfo())

	err = model.LoadFromFile(in.Params.ModelPath)
	if err != nil {
		return
	}

	filenames, err := model.Predict(&in.Params)
	if err != nil {
		return
	}
	model.Close()

	if in.Params.WithUpscale {
		err = Upscale(model, in, filenames)
		if err != nil {
			return
		}
	}

	return
}

func Upscale(model *sd.Model, in *InputData, filenames []string) (err error) {
	err = model.LoadUpscaleModel(in.Params.UpscalePath)
	if err != nil {
		return
	}
	defer model.CloseUpscaleModel()

	var total = len(filenames)
	for i, file := range filenames {
		var filenameIn = filepath.Join(in.Params.OutputDir, file)
		var filenameOut = filenameIn
		if !in.Params.DeleteUpscaled {
			filenameOut = filepath.Join(in.Params.OutputDir, "u-"+file)
		}

		fmt.Printf("\nUpscaling %d/%d: %s\n\n", i+1, total, file)

		err = model.UpscaleImage(filenameIn, filenameOut, 2)
		if err != nil {
			return
		}
	}

	return
}
