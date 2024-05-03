package generate

import (
	"fmt"
	"net/http"
	"os"
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

	model, err := InitModel(in)
	if err != nil {
		println(err.Error())
		return
	}
	defer model.Close()

	err = Run(model, in)
	if err != nil {
		println(err.Error())
		return
	}

	fmt.Printf("\nDONE\n")

	_ = c.JSON(http.StatusOK, "OK")
	return
}

func InitModel(in *InputData) (model *sd.Model, err error) {
	model, err = sd.NewModel(&in.Options)
	if err != nil {
		return
	}

	// println(model.GetSystemInfo())

	err = model.LoadFromFile(in.Params.ModelPath)
	if err != nil {
		return
	}

	return
}

func Run(model *sd.Model, in *InputData) (err error) {
	filenames, err := model.Predict(&in.Params)
	if err != nil {
		return
	}

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

	for _, file := range filenames {
		var filenameIn = filepath.Join(in.Params.OutputDir, file)
		var filenameOut = filepath.Join(in.Params.OutputDir, "u-"+file)

		err = model.UpscaleImage(filenameIn, filenameOut, 2)
		if err != nil {
			return
		}

		if in.Params.DeleteUpscaled {
			err = os.Remove(filenameIn)
			if err != nil {
				return
			}
		}
	}

	return
}
