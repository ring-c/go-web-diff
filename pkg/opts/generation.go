package opts

import (
	"math/rand"
	"sync"
)

type Generation struct {
	Width     int
	Height    int
	Seed      int64
	OutputDir string

	fileWrite     sync.WaitGroup
	filenamesLock sync.RWMutex
	filenames     []string
}

func NewGeneration(in *Options) *Generation {
	var generation = Generation{
		Width:     in.Width,
		Height:    in.Height,
		Seed:      in.Seed,
		OutputDir: in.OutputDir,

		filenames: make([]string, 0),
	}

	if generation.Seed == -1 {
		generation.Seed = rand.Int63()
	}

	generation.fileWrite.Add(in.BatchCount)

	return &generation
}

func (gen *Generation) GetFilenames() []string {
	gen.fileWrite.Wait()

	gen.filenamesLock.Lock()
	defer gen.filenamesLock.Unlock()

	return gen.filenames
}

func (gen *Generation) OneDone() {
	gen.fileWrite.Done()
}

func (gen *Generation) AddFilename(filename string) {
	gen.filenamesLock.Lock()
	defer gen.filenamesLock.Unlock()

	gen.filenames = append(gen.filenames, filename)
}
