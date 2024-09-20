package sd

import (
	"unsafe"

	"github.com/ring-c/go-web-diff/pkg/opts"
)

func (sd *Model) ResultCallback(num uint64, imageData *byte, in *opts.Options) {
	var imgData = unsafe.Slice(imageData, in.Width*in.Height*3)
	go sd.writeFile(imgData, in, num, 777)
}
