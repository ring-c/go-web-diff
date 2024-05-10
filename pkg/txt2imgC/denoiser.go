package sd

import (
	"math"
)

var denoiser = denoiserStruct{}

type denoiserStruct struct {
}

func (dn denoiserStruct) GetSigmas(steps int) (sigmas []float64) {
	sigmas = make([]float64, 0)

	var sigmaMin = 0.1
	var sigmaMax = 10.0
	var rho = 7.0

	var minInvRho = math.Pow(sigmaMin, 1.0/rho)
	var maxInvRho = math.Pow(sigmaMax, 1.0/rho)

	var stepsF = (float64(steps) - 1) * (minInvRho - maxInvRho)
	for i := 0; i < steps; i++ {
		var result = math.Pow(maxInvRho+float64(i)/stepsF, rho)
		sigmas = append(sigmas, result)
	}
	sigmas = append(sigmas, 0.0)

	return
}
