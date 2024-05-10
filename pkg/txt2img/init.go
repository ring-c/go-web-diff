package main

import (
	"fmt"

	"github.com/davecgh/go-spew/spew"
	"gorgonia.org/tensor"
)

func main() {
	spew.Config.Indent = "\t"

	a := tensor.New(tensor.WithBacking(tensor.Range(tensor.Float32, 0, 192)), tensor.WithShape(8, 8, 3))

	fmt.Printf("a:\n%v\n", a)

	data, err := tensorToImage(a)
	if err != nil {
		panic(err.Error())
		return
	}

	spew.Dump(data)
}
