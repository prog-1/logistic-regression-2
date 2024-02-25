package main

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	/*
		True positives: p: 0.7 | y: 1
		True negatives: p: 0.3 | y: 0
		False positives: p: 0.3 | y: 1
		False negatives: p: 0.7 | y: 0

		Accuracy  = (True Positives + True Negatives)/(True Positives + True Negatives + False Positives + False Negatives)
	*/

	predictions := inference(inputs, w, b) // getting predictions of logistics regression

	// ##################### Calculating counters #######################

	var truePositives, trueNegatives, falsePositives, falseNegatives float64

	for i, p := range predictions {
		if p >= 0.5 { // prediction is 1
			if y[i] == 1 { // truth is 1
				truePositives++
			} else { // truth is 0
				falsePositives++
			}
		} else { //prediction is 0
			if y[i] == 1 { // truth is 1
				falseNegatives++
			} else { // truth is 0
				trueNegatives++
			}
		}
	}

	// ##################### Formula #######################

	return (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)

}
