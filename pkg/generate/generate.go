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

	model, err := sd.NewModel(&in.Options)
	if err != nil {
		println(err.Error())
		return
	}
	defer model.Close()

	// println(model.GetSystemInfo())

	err = model.LoadFromFile(in.Params.ModelPath)
	if err != nil {
		println(err.Error())
		return
	}

	filenames, err := model.Predict(&in.Params)
	if err != nil {
		println(err.Error())
		return
	}

	if in.Params.WithUpscale {
		err = upscale(model, in.Params.UpscalePath, in.Params.DeleteUpscaled, filenames)
		if err != nil {
			println(err.Error())
			return
		}
	}

	fmt.Printf("\nDONE\n")

	_ = c.JSON(http.StatusOK, "OK")
	return
}

func upscale(model *sd.Model, path string, deleteUpscaled bool, filenames []string) (err error) {
	model.LoadUpscaleModel(path)
	defer model.CloseUpscaleModel()

	var outputDir = "./output/"

	for _, file := range filenames {
		var filename = filepath.Join(outputDir, file)
		var filenameOut = filepath.Join(outputDir, "u-"+file)

		var fileRead *os.File
		fileRead, err = os.Open(filename)
		if err != nil {
			return
		}

		defer func() {
			_ = fileRead.Close()
		}()

		var fileWrite *os.File
		fileWrite, err = os.Create(filenameOut)
		if err != nil {
			return
		}

		defer func() {
			_ = fileWrite.Close()
		}()

		err = model.UpscaleImage(fileRead, 2, fileWrite)
		if err != nil {
			return
		}

		if deleteUpscaled {
			err = os.Remove(filename)
			if err != nil {
				return
			}
		}
	}

	return
}
