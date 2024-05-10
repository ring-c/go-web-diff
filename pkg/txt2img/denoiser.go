package main

import (
	"math"

	"gorgonia.org/tensor"
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

func (dn denoiserStruct) DeNoise(input *tensor.Dense, sigma float64, step int) {
	cSkip := 1.0
	cOut, cIn := dn.GetScaling(sigma)

	t := denoiser.SigmaToT(sigma)
	timestepsVec := make([]float32, x.Ne[3])
	for i := range timestepsVec {
		timestepsVec[i] = t
	}
	timesteps := vectorTotensor.Dense(workCtx, timestepsVec)

	copytensor.Dense(noisedInput, input)
	tensor.DenseScale(noisedInput, cIn)

	var controls []*tensor.Dense

	if startMergeStep == -1 || step <= startMergeStep {
		diffusionModel.Compute(nThreads, noisedInput, timesteps, c, cConcat, cVector, -1, controls, controlStrength, &outCond)
	} else {
		diffusionModel.Compute(nThreads, noisedInput, timesteps, cID, cConcat, cVecID, -1, controls, controlStrength, &outCond)
	}

	var negativeData []float32
	if hasUnconditioned {
		if controlHint != nil {
			controlNet.Compute(nThreads, noisedInput, controlHint, timesteps, uc, ucVector)
			controls = controlNet.Controls
		}
		diffusionModel.Compute(nThreads, noisedInput, timesteps, uc, ucConcat, ucVector, -1, controls, controlStrength, &outUncond)
		negativeData = outUncond.Data
	}
	vecDenoised := denoised.Data
	vecInput := input.Data
	positiveData := outCond.Data
	neElements := ggmlNelements(denoised)
	for i := 0; i < neElements; i++ {
		latentResult := positiveData[i]
		if hasUnconditioned {
			ne3 := outCond.Ne[3]
			if minCfg != cfgScale && ne3 != 1 {
				i3 := i / (outCond.Ne[0] * outCond.Ne[1] * outCond.Ne[2])
				scale := minCfg + (cfgScale-minCfg)*float32(i3)/float32(ne3)
			} else {
				latentResult = negativeData[i] + cfgScale*(positiveData[i]-negativeData[i])
			}
		}
		vecDenoised[i] = latentResult*cOut + vecInput[i]*cSkip
	}
}

func (dn denoiserStruct) GetScaling(sigma float64) (float64, float64) {
	var cIn = 1.0 / math.Sqrt(sigma*sigma+1*1)
	return -sigma, cIn
}
