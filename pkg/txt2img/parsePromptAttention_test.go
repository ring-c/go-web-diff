package txt2img

import (
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func Test_parsePromptAttention(t *testing.T) {

	var resp = parsePromptAttention("(cat:2) dog")

	spew.Dump(resp)

}
