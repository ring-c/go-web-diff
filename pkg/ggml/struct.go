package ggml

import (
	"unsafe"
)

type Struct struct {
	InitGo         func(memSize uint64) unsafe.Pointer
	TensorOverhead func() uint64

	NewTensor4D  func(upscaleCTX unsafe.Pointer, ggmlType, imgWidth, imgHeight, dim, size int) unsafe.Pointer
	TensorSetF32 func(tx unsafe.Pointer, value float64, l, k, j, i int)
}

type InitParams struct {
	// memory pool
	MemSize   uint64 // bytes
	MemBuffer []byte // if nil, memory will be allocated internally
	NoAlloc   bool   // don't allocate memory for the tensor data
}
