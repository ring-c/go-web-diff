package opts

type Params struct {
	ModelPath string `json:"model_path"`

	Prompt           string           `json:"prompt"`
	NegativePrompt   string           `json:"negative_prompt"`
	ClipSkip         int              `json:"clip_skip"`
	CfgScale         float32          `json:"cfg_scale"`
	Width            int              `json:"width"`
	Height           int              `json:"height"`
	SampleMethod     SampleMethod     `json:"sample_method"`
	SampleSteps      int              `json:"sample_steps"`
	Strength         float32          `json:"strength"`
	Seed             int64            `json:"seed"`
	BatchCount       int              `json:"batch_count"`
	OutputsImageType OutputsImageType `json:"outputs_image_type"`
}
