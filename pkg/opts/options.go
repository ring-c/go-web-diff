package opts

type Options struct {
	VaePath               string
	TaesdPath             string
	LoraModelDir          string
	VaeDecodeOnly         bool
	VaeTiling             bool
	FreeParamsImmediately bool
	Threads               int16
	Wtype                 WType
	RngType               RNGType
	Schedule              Schedule
	GpuEnable             bool
	Debug                 bool
}
