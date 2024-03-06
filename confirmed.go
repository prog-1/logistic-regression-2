package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
func Dot(a []float64, b []float64) (dot float64) {
	if len(a) != len(b) {
		fmt.Println(a, b)
		panic("len(a) != len(b)")
	}
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return dot
}
func Inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	res = make([]float64, len(inputs))
	for i := 0; i < len(inputs); i++ {
		res[i] = Sigmoid(Dot(inputs[i], w) + b)
	}
	return res
}
func DCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(polynomial(0, 0)))
	m := len(inputs)
	for i := 0; i < m; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += (inputs[i][j] * (p[i] - y[i])) / float64(m)
		}
		db += (p[i] - y[i]) / float64(m)
	}
	return dw, db
}
func GradientDescent(inputs [][]float64, y, w, geta []float64, getab, alpha, b float64, epochs int) ([]float64, float64, []float64, float64) {
	var dw []float64
	var db float64
	for i := 0; i < epochs; i++ {
		p := Inference(inputs, w, b)
		dw, db = DCost(inputs, y, p)
		for j := 0; j < len(w); j++ {
			geta[j] += dw[j] * dw[j]
			w[j] -= alpha / math.Sqrt(geta[j]+1e-8) * dw[j]
		}
		getab += db * db
		b -= alpha / math.Sqrt(getab+1e-8) * db
		// if i%100000 == 0 {
		// 	fmt.Printf("Epoch: %v, %v, %v \n", i, dw, db)
		// }
	}
	return w, b, dw, db
}
func Accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	prediction := Inference(inputs, w, b)
	var truePos, trueNeg, falsePos, falseNeg float64
	for i := 0; i < len(y); i++ {
		if prediction[i] >= 0.5 {
			if y[i] == 1 {
				truePos++
			} else {
				falsePos++
			}
		} else {
			if y[i] == 0 {
				trueNeg++
			} else {
				falseNeg++
			}
		}
	}
	return (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
}
func Split(data [][]string) (xTrain, xTest, kapusta [][]float64, yTrain, yTest []float64) {
	half, segment := len(data)/2, len(data[0])

	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	xTrain, xTest = make([][]float64, half), make([][]float64, half)
	for i := range xTrain {
		xTrain[i] = make([]float64, segment-1)
		xTest[i] = make([]float64, segment-1)
	}
	yTrain, yTest = make([]float64, half), make([]float64, half)

	for i, row := range data[:half] {
		for j := 0; j < 2; j++ {
			xTrain[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTrain[i], _ = strconv.ParseFloat(row[2], 64)
	}
	for i, row := range data[half:] {
		for j := 0; j < 2; j++ {
			xTest[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		yTest[i], _ = strconv.ParseFloat(row[2], 64)
	}

	kapusta = make([][]float64, len(data))
	for i := range data {
		kapusta[i] = make([]float64, segment+1)
	}
	for i := range data {
		kapusta[i][0], _ = strconv.ParseFloat(data[i][0], 64)
		kapusta[i][1], _ = strconv.ParseFloat(data[i][1], 64)
	}
	return xTrain, xTest, kapusta, yTrain, yTest
}
func ReadData(adress string) (data [][]string) {
	file, err := os.Open(adress)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	data, err = reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	return data
}

type Plottable struct {
	//grid [][]float64
	N int
	M int
	f func(c, r int) float64
}

func (p Plottable) Dims() (c, r int) {
	return p.N, p.M
}
func (p Plottable) X(c int) float64 {
	return float64(c)
}
func (p Plottable) Y(r int) float64 {
	return float64(r)
}
func (p Plottable) Z(c, r int) float64 {
	return p.f(c, r)
}
