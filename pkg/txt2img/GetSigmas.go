package txt2img

import (
	"math"
)

type SigmaSchedule struct {
	AlphasCumprod []float32
	Sigmas        []float32
	LogSigmas     []float32
	Version       int
}

func (s *SigmaSchedule) GetSigmas(n uint32) []float32 {
	return s.Sigmas[:n]
}

func (s *SigmaSchedule) SigmaToT(sigma float32) float32 {
	logSigma := float32(math.Log(float64(sigma)))
	dists := make([]float32, len(s.LogSigmas))
	for i, logSigmaVal := range s.LogSigmas {
		dists[i] = logSigma - logSigmaVal
	}

	lowIdx := 0
	for i := range dists {
		if dists[i] >= 0 {
			lowIdx++
		}
	}
	lowIdx = int(math.Max(float64(math.Min(float64(lowIdx-1), 0)), float64(len(s.LogSigmas)-2)))
	highIdx := lowIdx + 1

	low := s.LogSigmas[lowIdx]
	high := s.LogSigmas[highIdx]
	w := (low - logSigma) / (low - high)
	w = float32(math.Max(0, math.Min(1, float64(w))))
	t := (1.0-float64(w))*float64(lowIdx) + float64(w)*float64(highIdx)

	return float32(t)
}

func (s *SigmaSchedule) TToSigma(t float32) float32 {
	lowIdx := int(math.Floor(float64(t)))
	highIdx := int(math.Ceil(float64(t)))
	w := t - float32(lowIdx)
	logSigma := (1.0-float64(w))*float64(s.LogSigmas[lowIdx]) + float64(w)*float64(s.LogSigmas[highIdx])
	return float32(math.Exp(logSigma))
}
