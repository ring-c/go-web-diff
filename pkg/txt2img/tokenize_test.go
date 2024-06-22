package txt2img

import (
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func Test_tokenize(t *testing.T) {
	tokens, weights := tokenize("a 1dog running on grass field", false)

	println("DONE")
	spew.Dump(tokens)
	spew.Dump(weights)
}
