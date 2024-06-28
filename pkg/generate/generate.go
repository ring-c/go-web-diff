package generate

import (
	"fmt"
	"net/http"
	"path/filepath"
	"sync"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
	"github.com/ring-c/go-web-diff/pkg/txt2img"
)

var (
	generator *txt2img.Generator
	lastModel string
)

type generateResp struct {
	Filenames []string `json:"filenames,omitempty"`
	Error     string   `json:"error,omitempty"`
}

func Generate(c echo.Context) (err error) {
	defer func() {
		if err != nil {
			println()
			fmt.Printf("\n\nERROR: %s \n\n", err.Error())
		}
	}()

	var resp = generateResp{
		Filenames: make([]string, 0),
	}

	in, err := getInput(c)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	if in.ModelPath != lastModel {
		generator, err = txt2img.New(in)
		if err != nil {
			fmt.Printf("\n\n\nERROR INIT\n%s\n\n\n", err.Error())
			return
		}
		lastModel = in.ModelPath
		generator.ApplyLoras(generator.Model.GetCTX())
	}

	resp.Filenames, err = generator.Generate(in)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	/*
		if in.WithUpscale {
			err = Upscale(model, in, filenames)
			if err != nil {
				return
			}
		}
	*/

	_ = c.JSON(http.StatusOK, resp)
	return
}

func ModelClose() {
	generator.Model.Close()
}

func Upscale(model *sd.Model, in *opts.Options, filenames []string) (err error) {
	err = model.LoadUpscaleModel(in.UpscalePath)
	if err != nil {
		return
	}
	defer model.CloseUpscaleModel()

	var total = len(filenames)
	var wg = new(sync.WaitGroup)
	for i, file := range filenames {
		var filenameIn = filepath.Join(in.OutputDir, file)
		var filenameOut = filenameIn
		if !in.DeleteUpscaled {
			filenameOut = filepath.Join(in.OutputDir, "u-"+file)
		}

		if in.Debug {
			fmt.Printf("\nUpscaling %d/%d: %s\n\n", i+1, total, file)
		}

		err = model.UpscaleImage(wg, filenameIn, filenameOut, 2)
		if err != nil {
			return
		}
	}

	wg.Wait()
	return
}
