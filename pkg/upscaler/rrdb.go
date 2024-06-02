package upscaler

import (
	"strconv"

	"github.com/ggml/ggml"
)

type RRDBNet struct {
	scale       int
	num_block   int
	num_in_ch   int
	num_out_ch  int
	num_feat    int
	num_grow_ch int
	blocks      map[string]ggml.Block
}

func NewRRDBNet() *RRDBNet {
	r := &RRDBNet{
		scale:       4,
		num_block:   6,
		num_in_ch:   3,
		num_out_ch:  3,
		num_feat:    64,
		num_grow_ch: 32,
		blocks:      make(map[string]ggml.Block),
	}

	r.blocks["conv_first"] = ggml.NewConv2d(r.num_in_ch, r.num_feat, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})
	for i := 0; i < r.num_block; i++ {
		name := "body." + strconv.Itoa(i)
		r.blocks[name] = ggml.NewRRDB(r.num_feat, r.num_grow_ch)
	}
	r.blocks["conv_body"] = ggml.NewConv2d(r.num_feat, r.num_feat, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})
	r.blocks["conv_up1"] = ggml.NewConv2d(r.num_feat, r.num_feat, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})
	r.blocks["conv_up2"] = ggml.NewConv2d(r.num_feat, r.num_feat, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})
	r.blocks["conv_hr"] = ggml.NewConv2d(r.num_feat, r.num_feat, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})
	r.blocks["conv_last"] = ggml.NewConv2d(r.num_feat, r.num_out_ch, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1})

	return r
}

func (r *RRDBNet) Lrelu(ctx *ggml.Context, x ggml.Tensor) ggml.Tensor {
	return ggml.LeakyRelu(ctx, x, 0.2, true)
}

func (r *RRDBNet) Forward(ctx *ggml.Context, x ggml.Tensor) ggml.Tensor {
	convFirst := r.blocks["conv_first"].(ggml.Conv2d)
	convBody := r.blocks["conv_body"].(ggml.Conv2d)
	convUp1 := r.blocks["conv_up1"].(ggml.Conv2d)
	convUp2 := r.blocks["conv_up2"].(ggml.Conv2d)
	convHr := r.blocks["conv_hr"].(ggml.Conv2d)
	convLast := r.blocks["conv_last"].(ggml.Conv2d)

	feat := convFirst.Forward(ctx, x)
	bodyFeat := feat
	for i := 0; i < r.num_block; i++ {
		name := "body." + strconv.Itoa(i)
		block := r.blocks[name].(ggml.RRDB)
		bodyFeat = block.Forward(ctx, bodyFeat)
	}
	bodyFeat = convBody.Forward(ctx, bodyFeat)
	feat = ggml.Add(ctx, feat, bodyFeat)

	feat = r.Lrelu(ctx, convUp1.Forward(ctx, ggml.Upscale(ctx, feat, 2)))
	feat = r.Lrelu(ctx, convUp2.Forward(ctx, ggml.Upscale(ctx, feat, 2)))
	out := convLast.Forward(ctx, r.Lrelu(ctx, convHr.Forward(ctx, feat)))
	return out
}
