package main

import "math"

//####################################################################################################################

// Converts 2D slice of x0 & x1 into 2D slice of the given power
func convert2Ds(x [][]float64) [][]float64 {
	res := make([][]float64, len(x))
	for i, p := range x { //p = point
		res[i] = convert(p)
	}
	return res
}

/*
	0 : x0
	1 : x1
	2 : x0*x0
	3 : x1*x1
	4 : x0*x1
*/
// Converts slice of x0 & x1 into slice of the given power
func convert(x []float64) []float64 {

	res := []float64{x[0], x[1]}

	for i := 0; i <= power; i++ {
		for j := 0; j <= power-i; j++ {
			if (i == 0 && j == 0) || (i == 1 && j == 0) || (i == 0 && j == 1) {
				continue
			}
			res = append(res, math.Pow(x[0], float64(i))*math.Pow(x[1], float64(j)))
		}
	}

	return res
}
