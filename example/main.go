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
	inputFileName = "data/two_circles.csv"
	epochs        = 1e+5
	learningRateW = 1e-9
	learningRateB = 3e-1
	//cubic: arcs/two_circles/circle/exams   blobs:
	//lrW, lrB = 1e-9, 3e-1                   1e-7, 4e-1
	//quadratic two_circles:
	//lrW, lrW =1e-7, 4e-1
	//linear:  arcs/blobs/circle/     two_circles:
	//lrW, lrB = 1e-3, 2e-1            1e-8, 2e-1
)

var (
	rnd = rand.New(rand.NewSource(10))
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
	var maxX, maxY float64
	for i := range inputs {
		if inputs[i][0] > maxX {
			maxX = inputs[i][0]
		}
		if inputs[i][1] > maxY {
			maxY = inputs[i][1]
		}
	}
	inputs, y = shuffle(inputs, y)
	pow := 3
	pInputs := polynomial(inputs, pow)
	pointPlt := dataPlot{
		posShape:   draw.PlusGlyph{},
		negShape:   draw.RingGlyph{},
		defaultClr: color.RGBA{A: 255},
		trueClr:    color.RGBA{G: 255, A: 255},
		falseClr:   color.RGBA{R: 255, A: 255},
	}
	xTrain, xTest, yTrain, yTest := split(pInputs, y)
	for i := range yTrain {
		pointPlt.addTrain(xTrain[i][0], xTrain[i][1], yTrain[i])
	}
	w := make([]float64, len(pInputs[0]))
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
	fmt.Println("Accuracy:", accuracy(xTest, yTest, w, b))
	prob := inference(xTest, w, b)
	for i := range prob {
		pointPlt.addTest(xTest[i][0], xTest[i][1], yTest[i], prob[i])
	}
	boundPlot := decBoundPlot{
		rows: int(maxY + 1.5),
		cols: int(maxX + 1.5),
		f: func(c, r int) float64 {
			return p(pPolynomial([]float64{float64(c), float64(r)}, pow), w, b)
		},
	}
	plotters := []plot.Plotter{
		plotter.NewContour(boundPlot, []float64{0.5}, palette.Heat(1, 255)),
	}
	pps := pointPlt.series()
	for _, p := range pps {
		plotters = append(plotters, p)
	}
	legend := fmt.Sprintf("Accuracy: %.2f", accuracy(xTest, yTest, w, b))
	if err := ebiten.RunGame(&App{img: ebiten.NewImageFromImage(Plot(screenWidth, screenHeight, legend, plotters...))}); err != nil {
		log.Fatal(err)
	}
}

func polynomial(inputs [][]float64, pow int) (pInputs [][]float64) {
	pInputs = make([][]float64, len(inputs))
	for i := range inputs {
		pInputs[i] = pPolynomial([]float64{inputs[i][0], inputs[i][1]}, pow)
	}
	return pInputs
}

func pPolynomial(x []float64, pow int) (px []float64) {
	px = []float64{x[0], x[1]}
	for i := 0; i <= pow; i++ {
		for j := 0; j <= pow-i; j++ {
			if (i == 0 && j == 0) || (i == 1 && j == 0) || (i == 0 && j == 1) {
				continue
			}
			px = append(px, math.Pow(x[0], float64(i))*math.Pow(x[1], float64(j)))
		}
	}
	return px
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}
func dot(a []float64, b []float64) float64 {
	var res float64
	if len(a) != len(b) {
		return 0
	}
	for i := 0; i < len(a); i++ {
		res += a[i] * b[i]
	}
	return res
}
func p(x []float64, w []float64, b float64) float64 {
	return sigmoid(dot(w, x) + b)
}
func inference(inputs [][]float64, w []float64, b float64) []float64 {
	res := make([]float64, len(inputs))
	for j, x := range inputs {
		f := dot(w, x)
		res[j] = sigmoid(f + b)
	}
	return res
}
func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	m := float64(len(inputs))
	dw = make([]float64, len(inputs[0]))
	for i := range inputs {
		for j := range inputs[i] {
			dw[j] += (p[i] - y[i]) * inputs[i][j] / m
		}
		db += (p[i] - y[i]) / m
	}
	return dw, db
}
func accuracy(inputs [][]float64, y []float64, w []float64, b float64) (acc float64) {
	r := inference(inputs, w, b)
	for i := range inputs {
		if y[i] == math.Round(r[i]) {
			acc++
		}
	}
	return acc / float64(len(y))
}
func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	size := len(inputs) / 5
	return inputs[size:], inputs[:size], y[size:], y[:size]
}

func shuffle(inputs [][]float64, y []float64) (sInputs [][]float64, sY []float64) {
	ind := rnd.Perm(len(inputs))
	sInputs, sY = make([][]float64, len(inputs)), make([]float64, len(y))
	for i, j := range ind {
		sInputs[i], sY[i] = inputs[j], y[j]
	}
	return sInputs, sY
}
