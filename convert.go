package main

//####################################################################################################################

// Converts: [][] of x0 and x1 into: [][] of x0 and x1 in concrete power
func convert(x [][]float64, f func([]float64) []float64) [][]float64 {
	res := make([][]float64, len(x))
	for i, p := range x { //p = point
		res[i] = f(p)
	}
	return res
}

//при увеличении степерь необходимо указать все комбинации переменных, чтобы показатель степеней не превышал n

//w1*x0 + w2*x1 + w3*x0^2 + w4*x1^2 + w5*x0*x1 + b
func quadratic(x []float64) []float64 {
	return []float64{x[0], x[1], x[0] * x[0], x[1] * x[1], x[0] * x[1]}
}

//w1*x0 + w2*x1  + w3*x0^2 + w4*x1^2 + w5*x0*x1    + w6*x0^3 + w7*x1^3 + w8*x0^2*x1 + w9*x0*x1^2    + b
func cubic(x []float64) []float64 {
	return []float64{x[0], x[1], x[0] * x[0], x[1] * x[1], x[0] * x[1], x[0] * x[0] * x[0], x[1] * x[1] * x[1], x[0] * x[0] * x[1], x[0] * x[1] * x[1]}
}
