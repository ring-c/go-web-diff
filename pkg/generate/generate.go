package generate

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"

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

	err = model.Predict(&in.Params)
	if err != nil {
		println(err.Error())
		return
	}

	if in.Params.WithUpscale {
		err = upscale(in.Params.UpscalePath, model)
		if err != nil {
			println(err.Error())
			return
		}
	}

	println("DONE")

	_ = c.JSON(http.StatusOK, "OK")
	return
}

func upscale(path string, model *sd.Model) (err error) {
	model.LoadUpscaleModel(path)
	defer model.CloseUpscaleModel()

	var outputDir = "./output/"

	dir, err := os.ReadDir(outputDir)
	if err != nil {
		return
	}

	for _, file := range dir {
		if strings.HasPrefix(file.Name(), "u-") {
			continue
		}

		var filename = filepath.Join(outputDir, file.Name())
		var filenameOut = filepath.Join(outputDir, "u-"+file.Name())

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

		err = os.Remove(filename)
		if err != nil {
			return
		}
	}

	return
}
