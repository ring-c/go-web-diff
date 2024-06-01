package upscaler

import (
	"math"
)

// Tiling
func sdTiling(input, output *ggml.Tensor, scale, tileSize int, tileOverlapFactor float32, onTileProcess func(*ggml.Tensor, *ggml.Tensor, bool)) {
	inputWidth, inputHeight := int(input.Ne[0]), int(input.Ne[1])
	outputWidth, outputHeight := int(output.Ne[0]), int(output.Ne[1])
	if inputWidth%2 != 0 || inputHeight%2 != 0 || outputWidth%2 != 0 || outputHeight%2 != 0 {
		panic("input and output dimensions must be multiples of 2")
	}

	tileOverlap := int32(float32(tileSize) * tileOverlapFactor)
	nonTileOverlap := tileSize - tileOverlap

	params := ggml.InitParams{
		MemSize:   tileSize*tileSize*input.Ne[2]*4 + (tileSize*scale)*(tileSize*scale)*output.Ne[2]*4 + 3*ggml.TensorOverhead(),
		MemBuffer: nil,
		NoAlloc:   false,
	}
	logDebug("tile work buffer size: %.2f MB", float64(params.MemSize)/(1024*1024))

	// draft context
	tilesCtx := ggml.Init(params)
	if tilesCtx == nil {
		logError("ggml.Init() failed")
		return
	}
	defer ggml.Free(tilesCtx)

	// tiling
	inputTile := ggml.NewTensor4D(tilesCtx, ggml.TypeF32, tileSize, tileSize, input.Ne[2], 1)
	outputTile := ggml.NewTensor4D(tilesCtx, ggml.TypeF32, tileSize*scale, tileSize*scale, output.Ne[2], 1)
	onTileProcess(inputTile, nil, true)
	numTiles := int(math.Ceil(float64(inputWidth)/float64(nonTileOverlap)) * math.Ceil(float64(inputHeight)/float64(nonTileOverlap)))
	logInfo("processing %d tiles", numTiles)
	prettyProgress(1, numTiles, 0.0)
	tileCount := 1
	var lastY, lastX bool
	var lastTime float32
	for y := 0; y < inputHeight && !lastY; y += nonTileOverlap {
		if y+tileSize >= inputHeight {
			y = inputHeight - tileSize
			lastY = true
		}
		for x := 0; x < inputWidth && !lastX; x += nonTileOverlap {
			if x+tileSize >= inputWidth {
				x = inputWidth - tileSize
				lastX = true
			}
			t1 := ggml.TimeMs()
			ggml.SplitTensor2D(input, inputTile, x, y)
			onTileProcess(inputTile, outputTile, false)
			ggml.MergeTensor2D(outputTile, output, x*scale, y*scale, tileOverlap*scale)
			t2 := ggml.TimeMs()
			lastTime = float32((t2 - t1) / 1000.0)
			prettyProgress(tileCount, numTiles, lastTime)
			tileCount++
		}
		lastX = false
	}
	if tileCount < numTiles {
		prettyProgress(numTiles, numTiles, lastTime)
	}
}
