package bind

import (
	"github.com/davecgh/go-spew/spew"
)

func init() {
	spew.Config.Indent = "\t"
}

/*


func writeToFile(t *testing.T, byteData []byte, height int, width int, outputPath string) {

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := (y*width + x) * 3
			img.Set(x, y, color.RGBA{
				R: byteData[idx],
				G: byteData[idx+1],
				B: byteData[idx+2],
				A: 255,
			})
		}
	}

	file, err := os.Create(outputPath)
	if err != nil {
		t.Error(err)
	}
	defer func() {
		_ = file.Close()
	}()

	err = png.Encode(file, img)
	if err != nil {
		t.Error(err)
	}
	t.Log("Image saved at", outputPath)
}


func readFromFile(t *testing.T, path string) *sd.Image {
	file, err := os.Open(path)
	if err != nil {
		t.Error(err)
	}
	defer file.Close()
	decode, err := png.Decode(file)
	if err != nil {
		t.Error(err)
	}

	bounds := decode.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y
	size := width * height * 3
	img := make([]byte, size)
	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			idx := (y*width + x) * 3
			r, g, b, _ := decode.At(x, y).RGBA()
			img[idx] = byte(r)
			img[idx+1] = byte(g)
			img[idx+2] = byte(b)
		}
	}
	return &sd.Image{
		Width:  uint32(width),
		Height: uint32(height),
		Data:   img,
	}
}

func TestNewCStableDiffusionImg2Img(t *testing.T) {
	diffusion, err := sd.NewCStableDiffusion(getLibrary())
	if err != nil {
		t.Error(err)
		return
	}
	diffusion.SetLogCallBack(func(level sd.LogLevel, text string) {
		fmt.Printf("%s", text)
	})
	ctx := diffusion.NewCtx("./models/miniSD.ckpt", "", "", "", false, false, true, -1, sd.F16, sd.CUDA_RNG, sd.DEFAULT)
	defer diffusion.FreeCtx(ctx)

	img := readFromFile(t, "./assets/test.png")
	images := diffusion.ImagePredictImage(ctx, *img, "cat wears shoes, high quality", "", 0, 7.0, 256, 256, sd.EULER_A, 20, 0.4, 42, 1)

	writeToFile(t, images[0].Data, 256, 256, "./assets/test1.png")
}


func TestNewCStableDiffusionText2Img(t *testing.T) {
	var r = require.New(t)

	diffusion, err := NewCStableDiffusion()
	r.NoError(err)

	diffusion.SetLogCallBack(func(level opts.LogLevel, text string) {
		fmt.Printf("%s", text)
	})

	ctx := diffusion.NewCtx("../../models/dreamshaperXL_v21TurboDPMSDE.safetensors", "", "", "", false, false, true, 4, opts.F16, opts.CUDA_RNG, opts.DEFAULT)
	defer diffusion.FreeCtx(ctx)

	images := diffusion.PredictImage(ctx, "british short hair cat, high quality", "", 2, 2, 256, 256, opts.EULER_A, 4, 42, 1)

	writeToFile(t, images[0].Data, 256, 256, "../../output/test.png")
}
*/
