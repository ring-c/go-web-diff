package generate

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
	"github.com/ring-c/go-web-diff/pkg/txt2img"
)

var (
	model     *sd.Model
	generator *txt2img.Generator
	lastModel = ""
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

	var resp = generateResp{}

	in, err := getInput(c)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	if in.ModelPath != lastModel {
		println("generator init")
		generator, err = txt2img.New(in)
		if err != nil {
			fmt.Printf("\n\n\nERROR INIT\n%s\n\n\n", err.Error())
			return
		}
		lastModel = in.ModelPath
	}

	println("Generate")

	resp.Filenames, err = generator.Generate(in)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	if in.Debug {
		fmt.Printf("\nDONE\n\n")
	}

	_ = c.JSON(http.StatusOK, resp)
	return
}

func ModelClose() {
	generator.Model.Close()
}

func Run(in *opts.Options) (filenames []string, err error) {
	_ = os.Mkdir(in.OutputDir, os.ModePerm)
	if model == nil {
		model, err = sd.NewModel(in)
		if err != nil {
			return
		}
	}

	// println(model.GetSystemInfo())

	err = model.LoadFromFile(in.ModelPath)
	if err != nil {
		return
	}

	filenames, err = model.Predict(in)
	if err != nil {
		return
	}

	if in.WithUpscale {
		err = Upscale(model, in, filenames)
		if err != nil {
			return
		}
	}

	return
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
