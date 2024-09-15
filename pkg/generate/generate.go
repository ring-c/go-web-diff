package generate

import (
	"fmt"
	"net/http"
	"path/filepath"
	"sync"

	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/opts"
	"github.com/ring-c/go-web-diff/pkg/sd"
)

var (
	model *sd.Model

	lastModel string
	lastVAE   string
	lastLora  string
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

	if model == nil {
		model, err = sd.NewModel()
		if err != nil {
			fmt.Printf("\n\n\nERROR NewModel\n%s\n\n\n", err.Error())
			return
		}
	}

	model.SetOptions(in)

	if in.ModelPath != lastModel || in.VaePath != lastVAE || in.Lora != lastLora {
		err = model.LoadFromFile()
		if err != nil {
			fmt.Printf("\n\n\nERROR LoadFromFile\n%s\n\n\n", err.Error())
			return
		}

		lastModel = in.ModelPath
		lastVAE = in.VaePath
	}

	// if in.Lora != lastLora {
	// 	var loraApply = make([]string, 0)
	// 	for _, loraD := range strings.Split(in.Lora, ", ") {
	// 		var lora = strings.TrimSpace(loraD)
	// 		if lora == "" {
	// 			continue
	// 		}
	//
	// 		loraApply = append(
	// 			loraApply,
	// 			fmt.Sprintf("<lora:%s>", lora),
	// 		)
	// 	}
	//
	// 	generator.ApplyLora(generator.Model.GetCTX(), strings.Join(loraApply, ", "))
	lastLora = in.Lora
	// }

	resp.Filenames, err = model.Generate(in)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	if in.WithUpscale {
		err = Upscale(model, in, resp.Filenames)
		if err != nil {
			return
		}
	}

	_ = c.JSON(http.StatusOK, resp)
	return
}

func ModelClose() {
	model.Close()
}

func Upscale(model *sd.Model, in *opts.Options, filenames []string) (err error) {
	err = model.LoadUpscaleModel(in.UpscalePath)
	if err != nil {
		return
	}
	defer model.CloseUpscaleModel()

	var wg = new(sync.WaitGroup)
	for _, file := range filenames {
		var filenameIn = filepath.Join(in.OutputDir, file)
		var filenameOut = filenameIn
		if !in.DeleteUpscaled {
			filenameOut = filepath.Join(in.OutputDir, "u-"+file)
		}

		err = model.UpscaleImage(wg, filenameIn, filenameOut, 2)
		if err != nil {
			return
		}
	}

	wg.Wait()
	return
}
