package opts

type FullParams struct {
	Prompt           string
	NegativePrompt   string
	ClipSkip         int
	CfgScale         float32
	Width            int
	Height           int
	SampleMethod     SampleMethod
	SampleSteps      int
	Strength         float32
	Seed             int64
	BatchCount       int
	OutputsImageType OutputsImageType
}
