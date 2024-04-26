package opts

type Options struct {
	VaePath               string   `json:"vae_path"`                //
	TaesdPath             string   `json:"taesd_path"`              //
	LoraModelDir          string   `json:"lora_model_dir"`          //
	VaeDecodeOnly         bool     `json:"vae_decode_only"`         //
	VaeTiling             bool     `json:"vae_tiling"`              //
	FreeParamsImmediately bool     `json:"free_params_immediately"` //
	Threads               int16    `json:"threads"`                 // CPU generation
	WType                 WType    `json:"w_type"`                  //
	RngType               RNGType  `json:"rng_type"`                //
	Schedule              Schedule `json:"schedule"`                //
	GpuEnable             bool     `json:"gpu_enable"`              //
	Debug                 bool     `json:"debug"`                   //
}
