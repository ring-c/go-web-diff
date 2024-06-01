package upscaler

import (
	"unsafe"
)

type Upscaler struct {
	GGML GGMLStruct
}

type GGMLStruct struct {
	Init func(params GGMLInitParams) unsafe.Pointer
}

type GGMLInitParams struct {
	// memory pool
	MemSize   int64  // bytes
	MemBuffer []byte // if nil, memory will be allocated internally
	NoAlloc   bool   // don't allocate memory for the tensor data
}
