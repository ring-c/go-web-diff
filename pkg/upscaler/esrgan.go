package upscaler

import "log"

type ESRGAN struct {
	RRDBNet  RRDBNet
	Scale    int
	TileSize int
	// backend    ggml_backend_t
	// wtype      ggml_type
	// paramsCtx  *ggml_context
	// computeCtx *ggml_context
}

// backend ggml_backend_t, wtype ggml_type

func NewESRGAN() *ESRGAN {
	e := &ESRGAN{
		Scale:    4,
		TileSize: 128,
		// backend:  backend,
		// wtype:    wtype,
	}
	// e.rrdb_net.Init(e.paramsCtx, e.wtype)
	return e
}

func (e *ESRGAN) GetDesc() string {
	return "esrgan"
}

func (e *ESRGAN) LoadFromFile(filePath string) bool {
	log.Printf("loading esrgan from '%s'", filePath)

	e.allocParamsBuffer()
	esrganTensors := make(map[string]*ggml_tensor)
	e.RRDBNet.GetParamTensors(esrganTensors)

	modelLoader := NewModelLoader()
	if !modelLoader.InitFromFile(filePath) {
		log.Printf("init esrgan model loader from file failed: '%s'", filePath)
		return false
	}

	success := modelLoader.LoadTensors(esrganTensors, e.backend)
	if !success {
		log.Printf("load esrgan tensors from model loader failed")
		return false
	}

	log.Printf("esrgan model loaded")
	return success
}

/*
func (e *ESRGAN) BuildGraph(x *ggml_tensor) *ggml_cgraph {
	gf := ggml_new_graph(e.computeCtx)
	x = e.toBackend(x)
	out := e.rrdb_net.Forward(e.computeCtx, x)
	ggml_build_forward_expand(gf, out)
	return gf
}

func (e *ESRGAN) Compute(nThreads int, x *ggml_tensor, output **ggml_tensor, outputCtx *ggml_context) {
	getGraph := func() *ggml_cgraph {
		return e.BuildGraph(x)
	}
	GGMLModuleCompute(getGraph, nThreads, false, output, outputCtx)
}
*/
