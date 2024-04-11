package opts

type Options struct {
	VaePath               string
	TaesdPath             string
	LoraModelDir          string
	VaeDecodeOnly         bool
	VaeTiling             bool
	FreeParamsImmediately bool
	Threads               int
	Wtype                 WType
	RngType               RNGType
	Schedule              Schedule
	GpuEnable             bool
	Debug                 bool
}
