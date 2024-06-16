//go:build no

package upscaler

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/tidwall/gjson"
)

func strToGGMLType(dtype string) ggmlType {
	var ttype ggmlType = GGMLTypeCount
	switch dtype {
	case "F16":
		ttype = GGMLTypeF16
	case "BF16":
		ttype = GGMLTypeF32
	case "F32":
		ttype = GGMLTypeF32
	}
	return ttype
}

// https://huggingface.co/docs/safetensors/index
func (m *ModelLoader) initFromSafetensorsFile(filePath, prefix string) bool {
	fmt.Printf("init from '%s'\n", filePath)
	m.filePaths = append(m.filePaths, filePath)
	fileIndex := len(m.filePaths) - 1
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("failed to open '%s'\n", filePath)
		return false
	}
	defer file.Close()

	// get file size
	fileInfo, err := file.Stat()
	if err != nil {
		fmt.Printf("failed to get file info for '%s'\n", filePath)
		return false
	}
	fileSize := fileInfo.Size()

	// read header size
	if fileSize <= stHeaderSizeLen {
		fmt.Printf("invalid safetensor file '%s'\n", filePath)
		return false
	}

	headerSizeBuf := make([]byte, stHeaderSizeLen)
	_, err = io.ReadFull(file, headerSizeBuf)
	if err != nil {
		fmt.Printf("read safetensors header size failed: '%s'\n", filePath)
		return false
	}

	headerSize := binary.LittleEndian.Uint64(headerSizeBuf)
	if headerSize >= uint64(fileSize) {
		fmt.Printf("invalid safetensor file '%s'\n", filePath)
		return false
	}

	// read header
	headerBuf := make([]byte, headerSize)
	_, err = io.ReadFull(file, headerBuf)
	if err != nil {
		fmt.Printf("read safetensors header failed: '%s'\n", filePath)
		return false
	}

	header := gjson.Parse(string(headerBuf))

	for _, item := range header.Get("").Map() {
		name := item.Key
		if name == "__metadata__" {
			continue
		}
		if m.isUnusedTensor(name) {
			continue
		}

		dtype := item.Get("dtype").Str
		shape := item.Get("shape").Array()

		begin := item.Get("data_offsets.0").Uint()
		end := item.Get("data_offsets.1").Uint()

		ttype := strToGGMLType(dtype)
		if ttype == GGMLTypeCount {
			fmt.Printf("unsupported dtype '%s'\n", dtype)
			return false
		}

		nDims := len(shape)
		if nDims > sdMaxDims {
			fmt.Printf("invalid tensor '%s'\n", name)
			return false
		}

		ne := make([]int64, sdMaxDims)
		for i, s := range shape {
			ne[i] = s.Int()
		}

		if nDims == 5 {
			if ne[3] == 1 && ne[4] == 1 {
				nDims = 4
			} else {
				fmt.Printf("invalid tensor '%s'\n", name)
				return false
			}
		}

		// ggml_n_dims returns 1 for scalars
		if nDims == 0 {
			nDims = 1
		}

		tensorStorage := TensorStorage{
			name:       prefix + name,
			dtype:      ttype,
			ne:         ne,
			nDims:      nDims,
			fileIndex:  fileIndex,
			dataOffset: stHeaderSizeLen + uint64(headerSize) + begin,
		}
		tensorStorage.reverseNE()

		tensorDataSize := end - begin
		if dtype == "BF16" {
			tensorStorage.isBF16 = true
			GGMLASSERT(tensorStorage.nbytes() == tensorDataSize*2)
		} else {
			GGMLASSERT(tensorStorage.nbytes() == tensorDataSize)
		}

		m.tensorStorages = append(m.tensorStorages, tensorStorage)

		// fmt.Printf("%s %s\n", tensorStorage.toString(), dtype)
	}

	return true
}
