package sd

import (
	"time"
)

func denoise(input *ggmlTensor, sigma float32, step int) {

	t0 := time.Now()

	cSkip := 1.0
	cOut := 1.0
	cIn := 1.0
	scaling := denoiser.GetScalings(sigma)

	if len(scaling) == 3 { // CompVisVDenoiser
		cSkip = scaling[0]
		cOut = scaling[1]
		cIn = scaling[2]
	} else { // CompVisDenoiser
		cOut = scaling[0]
		cIn = scaling[1]
	}

	t := denoiser.Schedule.SigmaToT(sigma)
	timestepsVec := make([]float32, x.Ne[3])
	for i := range timestepsVec {
		timestepsVec[i] = t
	}
	timesteps := vectorToGgmlTensor(workCtx, timestepsVec)

	copyGgmlTensor(noisedInput, input)
	ggmlTensorScale(noisedInput, cIn)

	var controls []*ggmlTensor

	if controlHint != nil {
		controlNet.Compute(nThreads, noisedInput, controlHint, timesteps, c, cVector)
		controls = controlNet.Controls
	}

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
	t1 := time.Now()
	if step > 0 {
		prettyProgress(step, int(steps), float32(t1.Sub(t0).Microseconds())/1000000.0)
	}
}
