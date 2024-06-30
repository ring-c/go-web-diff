//go:build tests

package txt2img

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	UNK_TOKEN_ID int = 49407
	BOS_TOKEN_ID int = 49406
	EOS_TOKEN_ID int = 49407
	PAD_TOKEN_ID int = 49407
)

func (gen *Generator) Tokenize(text string, padding bool) (tokens []int, weights []float32) {
	tokens = make([]int, 0)
	weights = make([]float32, 0)

	for _, item := range parsePromptAttention(text) {

		var currText = item.Text
		var currWeight = item.Value

		currTokens := encode(currText)
		tokens = append(tokens, currTokens...)
		for range currTokens {
			weights = append(weights, currWeight)
		}
	}

	// padTokens(&tokens, &weights, maxLength, padding)

	return
}

func getFullPath(dir, filename string) string {
	dp, err := os.Open(dir)
	if err != nil {
		return ""
	}
	defer dp.Close()

	entries, err := dp.Readdir(-1)
	if err != nil {
		return ""
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.EqualFold(entry.Name(), filename) {
			return filepath.Join(dir, entry.Name())
		}
	}

	return ""
}

func padTokens(tokens []int, weights []float64, maxLength int, padding bool) ([]int, []float64) {
	if maxLength > 0 && padding {
		n := int(math.Ceil(float64(len(tokens)) / float64(maxLength-2)))
		if n == 0 {
			n = 1
		}
		length := maxLength * n
		fmt.Printf("token length: %d\n", length)

		newTokens := make([]int, 0, length+2)
		newWeights := make([]float64, 0, length+2)

		newTokens = append(newTokens, BOS_TOKEN_ID)
		newWeights = append(newWeights, 1.0)

		tokenIdx := 0
		for i := 1; i < length; i++ {
			if tokenIdx >= len(tokens) {
				break
			}
			if i%maxLength == 0 {
				newTokens = append(newTokens, BOS_TOKEN_ID)
				newWeights = append(newWeights, 1.0)
			} else if i%maxLength == maxLength-1 {
				newTokens = append(newTokens, EOS_TOKEN_ID)
				newWeights = append(newWeights, 1.0)
			} else {
				newTokens = append(newTokens, tokens[tokenIdx])
				newWeights = append(newWeights, weights[tokenIdx])
				tokenIdx++
			}
		}

		newTokens = append(newTokens, EOS_TOKEN_ID)
		newWeights = append(newWeights, 1.0)

		if padding {
			padTokenID := PAD_TOKEN_ID
			// if version == VERSION_2_X {
			// 	padTokenID = 0
			// }
			newTokens = append(newTokens, make([]int, length-len(newTokens), padTokenID)...)
			newWeights = append(newWeights, make([]float64, length-len(newWeights), 1.0)...)
		}

		return newTokens, newWeights
	}
	return tokens, weights
}
