package ggml

import (
	"unsafe"
)

type Struct struct {
	Init           func(memSize uint64) (ctx unsafe.Pointer)
	Free           func(ctx unsafe.Pointer)
	TensorOverhead func() uint64

	NewTensor4D      func(workCtx unsafe.Pointer, ggmlType, imgWidth, imgHeight, dim, size int) unsafe.Pointer
	TensorSetF32     func(tx unsafe.Pointer, value float64, l, k, j, i int)
	TensorSetF32Rand func(tx unsafe.Pointer, seed uint64)

	TensorScale       func(tx unsafe.Pointer, scale float32)
	TensorScaleOutput func(tx unsafe.Pointer)
	TensorClamp       func(tx unsafe.Pointer, min float32, max float32)

	TensorGetF32 func(tx unsafe.Pointer, l, k, j, i int) float32
	// VectorToGgmlTensorI32 func(workCtx unsafe.Pointer, vector unsafe.Pointer) unsafe.Pointer
}

type InitParams struct {
	// memory pool
	MemSize   uint64 // bytes
	MemBuffer []byte // if nil, memory will be allocated internally
	NoAlloc   bool   // don't allocate memory for the tensor data
}
