package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/labstack/echo/v4"

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

func init() {
	data, err := os.ReadFile("last_req.json")
	if err != nil {
		return
	}

	var in = DefaultInput
	err = json.Unmarshal(data, in)
	if err != nil {
		return
	}

	model, err = sd.NewModel()
	if err != nil {
		return
	}
	model.SetOptions(in)

	err = model.LoadFromFile()
	if err != nil {
		return
	}

	lastModel = in.ModelPath
	lastVAE = in.VaePath
	lastLora = in.Lora
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

	if in.WriteLastReq {
		data, _ := json.MarshalIndent(in, "", "\t")
		_ = os.WriteFile("last_req.json", data, 0644)
	}

	err = os.Mkdir(in.OutputDir, 0755)
	if err != nil {
		if !errors.Is(err, os.ErrExist) {
			resp.Error = err.Error()
			_ = c.JSON(http.StatusOK, resp)
			return
		}
		err = nil
	}

	if in.ReloadModel {
		ModelClose()
	}

	if model == nil {
		model, err = sd.NewModel()
		if err != nil {
			fmt.Printf("\n\n\nERROR NewModel\n%s\n\n\n", err.Error())
			return
		}
	}

	model.SetOptions(in)

	if in.ReloadModel || in.ModelPath != lastModel || in.VaePath != lastVAE || in.Lora != lastLora {
		err = model.LoadFromFile()
		if err != nil {
			fmt.Printf("\n\n\nERROR LoadFromFile\n%s\n\n\n", err.Error())
			return
		}

		lastModel = in.ModelPath
		lastVAE = in.VaePath
		lastLora = in.Lora
	}

	resp.Filenames, err = model.Txt2Img(in)
	if err != nil {
		resp.Error = err.Error()
		_ = c.JSON(http.StatusOK, resp)
		return
	}

	/*
		if in.WithUpscale {
			err = Upscale(model, in, resp.Filenames)
			if err != nil {
				return
			}
		}
	*/

	_ = c.JSON(http.StatusOK, resp)
	return
}

func ModelClose() {
	if model == nil {
		return
	}

	model.Close()
	model = nil
}

/*
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
*/