package main

import (
	"fmt"

	"gorgonia.org/tensor"
)

func main() {
	print("\n\n\n\n\n\n\n")

	a := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]int{1, 2, 3, 4}))
	fmt.Printf("a:\n%v\n", a)

}
