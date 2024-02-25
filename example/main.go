package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

type InputReader interface {
	Read() (x [][]float64, y []float64)
}

const (
	inputFileName = "C:/Common/Projects/School/logistic-regression-2/data/arcs.csv"
	// inputFileName = "C:/Common/Projects/School/logistic-regression-2/data/blobs.csv"
	// inputFileName = "C:/Common/Projects/School/logistic-regression-2/data/circle.csv"
	// inputFileName = "C:/Common/Projects/School/logistic-regression-2/data/exams.csv"
	// inputFileName = "C:/Common/Projects/School/logistic-regression-2/data/two_circles.csv"

	epochs        = 1e+6
	learningRateW = 1e-10
	learningRateB = 1e-1
)

var (
	rnd = rand.New(rand.NewSource(25))
)

func main() {
	const screenWidth, screenHeight = 640, 480
	ebiten.SetWindowSize(screenWidth, screenHeight)

	var (
		inputs [][]float64
		y      []float64
	)
	inputs, y, err := read(inputFileName)
	if err != nil {
		log.Fatal(err)
	}

	// inputs, w := linearInput(inputs)
	// inputs, w := quadraticInput(inputs)
	inputs, w := cubicInput(inputs)

	var maxX, maxY float64
	for i := range inputs {
		if inputs[i][0] > maxX {
			maxX = inputs[i][0]
		}
		if inputs[i][1] > maxY {
			maxY = inputs[i][1]
		}
	}

	pointPlt := dataPlot{
		posShape:   draw.PlusGlyph{},
		negShape:   draw.RingGlyph{},
		defaultClr: color.RGBA{A: 255},
		trueClr:    color.RGBA{G: 255, A: 255},
		falseClr:   color.RGBA{R: 255, A: 255},
	}
	xTrain, xTest, yTrain, yTest := split(inputs, y)
	for i := range yTrain {
		pointPlt.addTrain(xTrain[i][0], xTrain[i][1], yTrain[i])
	}

	// w := make([]float64, 2)
	var b float64
	for i := 0; i < epochs; i++ {
		p := inference(xTrain, w, b)
		dw, db := dCost(xTrain, yTrain, p)
		for i := range w {
			w[i] -= dw[i] * learningRateW
		}
		b -= db * learningRateB
	}
	fmt.Println("Weight:", w, "Bias:", b)
	score := accuracy(xTest, yTest, w, b)
	fmt.Println("Accuracy:", score)

	prob := inference(xTest, w, b)
	for i := range prob {
		pointPlt.addTest(xTest[i][0], xTest[i][1], yTest[i], prob[i])
	}

	boundPlot := decBoundPlot{
		rows: int(maxY + 1.5),
		cols: int(maxX + 1.5),
		f: func(c, r int) float64 {
			// return p(linearDecisionBoundaryInput(c, r), w, b)
			// return p(quadraticDecisionBoundaryInput(c, r), w, b)
			return p(cubicDecisionBoundaryInput(c, r), w, b)
		},
	}

	plotters := []plot.Plotter{
		plotter.NewContour(boundPlot, []float64{0.5}, palette.Heat(1, 255)),
	}
	pps := pointPlt.series()
	for _, p := range pps {
		plotters = append(plotters, p)
	}
	legend := fmt.Sprintf("Accuracy: %.2f", score)
	if err := ebiten.RunGame(&App{img: ebiten.NewImageFromImage(Plot(screenWidth, screenHeight, legend, plotters...))}); err != nil {
		log.Fatal(err)
	}
}

func linearInput(linearInput [][]float64) ([][]float64, []float64) {
	return linearInput, make([]float64, 2)
}

func linearDecisionBoundaryInput(c, r int) []float64 {
	return []float64{float64(c), float64(r)}
}

func quadraticInput(linearInput [][]float64) (quadraticInput [][]float64, w []float64) {
	quadraticInput = make([][]float64, len(linearInput))
	for i, x := range linearInput {
		if len(x) != 2 {
			panic("Inner slices must have length 2 in linear input")
		}
		quadraticInput[i] = []float64{x[0], x[1], x[0] * x[0], x[1] * x[1]}
	}
	return quadraticInput, make([]float64, 4)
}

func quadraticDecisionBoundaryInput(c, r int) []float64 {
	a, b := float64(c), float64(r)
	return []float64{a, b, a * a, b * b}
}

func cubicInput(linearInput [][]float64) (cubicInput [][]float64, w []float64) {
	cubicInput = make([][]float64, len(linearInput))
	for i, x := range linearInput {
		if len(x) != 2 {
			panic("Inner slices must have length 2 in linear input")
		}
		cubicInput[i] = polynomial(x[0], x[1], 3)
	}
	return cubicInput, make([]float64, len(cubicInput[0]))
}

func cubicDecisionBoundaryInput(c, r int) []float64 {
	a, b := float64(c), float64(r)
	return polynomial(a, b, 3)
}

func polynomial(x1, x2 float64, n int) (res []float64) {
	return []float64{x1, x2, math.Pow(x1, 3), math.Pow(x1, 2), math.Pow(x2, 3), math.Pow(x2, 2), x1 * x1 * x2, x1 * x2 * x2, x1 * x2}
	// for i := 0; i <= n; i++ {
	// 	for j := 0; j <= n; j++ {
	// 		if i+j <= n && i+j != 0 {
	// 			res = append(res, math.Pow(x1, float64(i))*math.Pow(x2, float64(j)))
	// 		}
	// 	}
	// }
	// return res
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	trainIndices := make(map[int]bool)
	for i := 0; i < len(inputs)/5*4; i++ {
		idx := rnd.Intn(len(inputs))
		for trainIndices[idx] {
			idx = rnd.Intn(len(inputs))
		}
		trainIndices[idx] = true
	}
	for i := 0; i < len(inputs); i++ {
		if trainIndices[i] {
			xTrain = append(xTrain, inputs[i])
			yTrain = append(yTrain, y[i])
		} else {
			xTest = append(xTest, inputs[i])
			yTest = append(yTest, y[i])
		}
	}
	return xTrain, xTest, yTrain, yTest
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	var res []float64
	for _, x := range inputs {
		res = append(res, p(x, w, b))
	}
	return res
}

func p(x []float64, w []float64, b float64) float64 {
	return sigmoid(dot(w, x) + b)
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func dot(a []float64, b []float64) (res float64) {
	if len(a) != len(b) {
		panic("Length of a and b must be equal")
	}
	for i := 0; i < len(a); i++ {
		res += a[i] * b[i]
	}
	return res
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	m := len(inputs)
	n := len(inputs[0])
	dw = make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dw[j] += (p[i] - y[i]) * inputs[i][j]
		}
		db += p[i] - y[i]
	}
	for j := 0; j < n; j++ {
		dw[j] /= float64(m)
	}
	db /= float64(m)
	return dw, db
}

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	p := inference(inputs, w, b)
	var trueOut float64
	for i := range p {
		if int(p[i]+0.5) == int(y[i]+0.5) {
			trueOut++
		}
	}
	return trueOut / float64(len(p))
}
