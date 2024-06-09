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

	// Ex Params
	ModelPath string `json:"model_path"`

	UpscalePath    string `json:"upscale_path"`
	WithUpscale    bool   `json:"with_upscale"`
	DeleteUpscaled bool   `json:"delete_upscaled"`

	OutputDir string `json:"output_dir"`

	Prompt           string           `json:"prompt"`
	NegativePrompt   string           `json:"negative_prompt"`
	ClipSkip         int              `json:"clip_skip"`
	CfgScale         float32          `json:"cfg_scale"`
	Width            int              `json:"width"`
	Height           int              `json:"height"`
	SampleMethod     SampleMethod     `json:"sample_method"`
	SampleSteps      int              `json:"sample_steps"`
	Seed             int64            `json:"seed"`
	BatchCount       int              `json:"batch_count"`
	OutputsImageType OutputsImageType `json:"outputs_image_type"`
}
