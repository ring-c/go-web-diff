package opts

type OutputsImageType string

const (
	PNG  OutputsImageType = "PNG"
	JPEG                  = "JPEG"
)

type SampleMethod uint8

const (
	EULER_A SampleMethod = iota
	EULER
	HEUN
	DPM2
	DPMPP2S_A
	DPMPP2M
	DPMPP2Mv2
	LCM
	N_SAMPLE_METHODS
)

type RNGType uint8

const (
	STD_DEFAULT_RNG RNGType = iota
	CUDA_RNG
)

type Schedule uint8

const (
	DEFAULT Schedule = iota
	DISCRETE
	KARRAS
	N_SCHEDULES
)

type WType uint8

const (
	F32   WType = 0
	F16         = 1
	Q4_0        = 2
	Q4_1        = 3
	Q5_0        = 6
	Q5_1        = 7
	Q8_0        = 8
	Q8_1        = 9
	Q2_K        = 10
	Q3_K        = 11
	Q4_K        = 12
	Q5_K        = 13
	Q6_K        = 14
	Q8_K        = 15
	I8          = 16
	I16         = 17
	I32         = 18
	COUNT       = 19 // don't use this when specifying a type
)
