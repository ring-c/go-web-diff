package opts

type Options struct {
	VaePath               string   `json:"vae_path"`                //
	TaesdPath             string   `json:"taesd_path"`              //
	LoraModelDir          string   `json:"lora_model_dir"`          //
	VaeDecodeOnly         bool     `json:"vae_decode_only"`         //
	VaeTiling             bool     `json:"vae_tiling"`              //
	FreeParamsImmediately bool     `json:"free_params_immediately"` //
	Threads               uint8    `json:"threads,string"`          // CPU generation
	WType                 WType    `json:"w_type"`                  //
	RngType               RNGType  `json:"rng_type"`                //
	Schedule              Schedule `json:"schedule"`                //
	GpuEnable             bool     `json:"gpu_enable"`              //
	Debug                 bool     `json:"debug"`                   //

	// Ex Params
	ModelPath string `json:"model_path"`

	UpscalePath    string `json:"upscale_path"`
	WithUpscale    bool   `json:"with_upscale"`
	DeleteUpscaled bool   `json:"delete_upscaled"`

	OutputDir string `json:"-"`

	Prompt           string           `json:"prompt"`
	NegativePrompt   string           `json:"negative_prompt"`
	ClipSkip         int              `json:"clip_skip,string"`
	CfgScale         float32          `json:"cfg_scale,string"`
	Width            int              `json:"width,string"`
	Height           int              `json:"height,string"`
	SampleMethod     SampleMethod     `json:"sample_method,string"`
	SampleSteps      int              `json:"sample_steps,string"`
	Seed             int64            `json:"seed,string"`
	BatchCount       int              `json:"batch_count,string"`
	OutputsImageType OutputsImageType `json:"outputs_image_type"`

	Lora string `json:"lora"`
}
