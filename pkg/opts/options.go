package opts

type Options struct {
	VaePath       string `json:"vae_path"`        //
	UseVae        bool   `json:"use_vae"`         //
	VaeDecodeOnly bool   `json:"vae_decode_only"` //
	VaeTiling     bool   `json:"vae_tiling"`      //

	// TaesdPath             string   `json:"taesd_path"`              //
	// Threads               uint8    `json:"threads,string"`          // CPU generation

	LoraModelDir          string   `json:"lora_model_dir"`          //
	FreeParamsImmediately bool     `json:"free_params_immediately"` //
	WType                 WType    `json:"w_type"`                  //
	RngType               RNGType  `json:"rng_type"`                //
	Schedule              Schedule `json:"schedule"`                //
	GpuEnable             bool     `json:"gpu_enable"`              //
	Debug                 bool     `json:"debug"`                   //
	ReloadModel           bool     `json:"reload_model"`            //

	// Ex Params
	ModelPath     string `json:"model_path"`
	FluxModelPath string `json:"flux_model_path"`
	ClipLPath     string `json:"clip_l_path"`
	T5xxlPath     string `json:"t5xxl_path"`

	UpscalePath    string `json:"upscale_path"`
	WithUpscale    bool   `json:"with_upscale"`
	DeleteUpscaled bool   `json:"delete_upscaled"`

	WithStepsPreview bool `json:"with_steps_preview"`

	OutputDir    string `json:"-"`
	WriteLastReq bool   `json:"write_last_req"`

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

	Lora      string `json:"lora"`
	ModelType int    `json:"model_type,string"`
}
