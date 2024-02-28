package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
func dot(a []float64, b []float64) (dot float64) {
	if len(a) != len(b) {
		fmt.Println(a, b)
		panic("len(a) != len(b)")
	}
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return dot
}
func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	res = make([]float64, len(inputs))
	for i := 0; i < len(inputs); i++ {
		res[i] = sigmoid(dot(inputs[i], w) + b)
	}
	return res
}
func quadraticInputX(inputs [][]float64) (xs [][]float64) {
	xs = inputs
	a, b := len(inputs), len(inputs[0])
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			xs[i] = append(xs[i], xs[i][j]*xs[i][j])
		}
	}
	return xs
}
func quadraticInputW(w []float64) (ws []float64) {
	ws = w
	c := len(w)
	for i := 0; i < c; i++ {
		ws = append(ws, ws[i]*ws[i])
	}
	return ws
}
func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
	m := len(inputs)
	for i := 0; i < m; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += (inputs[i][j] * (p[i] - y[i])) / float64(m)
		}
		db += (p[i] - y[i]) / float64(m)
	}
	return dw, db
}
func gradientDescent(inputs [][]float64, y, w []float64, alpha, b float64, epochs int) ([]float64, float64, []float64, float64) {
	var dw []float64
	var db float64
	for i := 0; i < epochs; i++ {
		p := inference(inputs, w, b)
		dw, db = dCost(inputs, y, p)
		for j := 0; j < len(w); j++ {
			w[j] -= alpha * dw[j]
		}
		b -= alpha * db
	}
	return w, b, dw, db
}
func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	//fmt.Println(inputs, y, w, b)
	prediction := inference(inputs, w, b)
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
func split(data [][]string) (xTrain, xTest, kapusta [][]float64, yTrain, yTest []float64) {
	half, segment := len(data)/2, len(data[0])
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
func drawLine(w []float64, b float64, p *plot.Plot) {
	line, _ := plotter.NewLine(plotter.XYs{
		{X: 0, Y: (-w[0]*0 - b) / w[1]},
		{X: 100, Y: (-w[0]*100 - b) / w[1]}})
	line.Color = color.RGBA{B: 255, A: 255}
	p.Add(line)
}
func drawQuadraticFunction(p *plot.Plot, f func(x float64) float64) {
	for i := 0; i < 1000; i++ {
		line, _ := plotter.NewLine(plotter.XYs{
			{X: float64(i), Y: f(float64(i))},
			{X: float64(i + 1), Y: f(float64(i + 1))}})
		line.Color = color.RGBA{G: 255, A: 255}
		p.Add(line)
	}
}
func readData(adress string) (data [][]string) {
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
func drawScatter(xTrain [][]float64, yTrain []float64, p *plot.Plot, scale float64) {
	var drawR, drawG plotter.XYs
	for i := 0; i < len(yTrain); i++ {
		if yTrain[i] == 0 {
			drawR = append(drawR, struct{ X, Y float64 }{X: xTrain[i][0] * scale, Y: xTrain[i][1] * scale})
		} else {
			drawG = append(drawG, struct{ X, Y float64 }{X: xTrain[i][0] * scale, Y: xTrain[i][1] * scale})
		}
	}
	scatterR, err := plotter.NewScatter(drawR)
	if err != nil {
		panic(err)
	}
	scatterR.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	scatterR.GlyphStyle.Radius = vg.Points(4)
	p.Add(scatterR)

	scatterG, err := plotter.NewScatter(drawG)
	if err != nil {
		panic(err)
	}
	scatterG.GlyphStyle.Color = color.RGBA{G: 255, A: 255}
	scatterG.GlyphStyle.Radius = vg.Points(4)
	p.Add(scatterG)

	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter.png"); err != nil {
		panic(err)
	}
}
func main() {
	//reading
	adress := "data/exams.csv"
	data := readData(adress)
	// variables
	p := plot.New()
	var dw []float64
	var db, scale float64
	xTrain, xTest, kapusta, yTrain, yTest := split(data)
	w := make([]float64, len(xTrain[0]))
	for i := range w {
		w[i] = rand.Float64() * 2
	}
	b := rand.Float64() * 2
	alpha := 1e-3
	epochs := 1000000
	// Output formatting
	fmt.Printf("Start values of weights and bias: %v, %v: \n", w, b)
	xTrain = quadraticInputX(xTrain)
	xTest = quadraticInputX(xTest)
	w = quadraticInputW(w)
	w, b, dw, db = gradientDescent(xTrain, yTrain, w, alpha, b, epochs)
	fmt.Printf("End values of weights and bias: %v, %v: \n", w, b)
	fmt.Printf("End values of dw and db: %v, %v: \n", dw, db)
	fmt.Printf("Epochs: %v\n", epochs)
	score := accuracy(xTest, yTest, w, b)
	fmt.Printf("Score: %v\n", score)
	// drawing
	var plotData plottable
	plotData.grid = kapusta
	plotData.N = len(kapusta)
	plotData.M = len(kapusta)
	plotData.f = func(c, r int) float64 {
		return sigmoid(dot([]float64{float64(c), float64(r), float64(c * c), float64(r * r)}, w) + b)
	}
	pal := moreland.SmoothBlueRed().Palette(255)
	heatmap := plotter.NewHeatMap(plotData, pal)
	p.Add(heatmap)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "heatmap.png"); err != nil {
		panic(err)
	}

	if adress == "data/exams.csv" {
		scale = 1
	} else if adress == "data/circle.csv" {
		scale = 100
	}
	drawScatter(xTrain, yTrain, p, scale)
	if adress != "data/exams.csv" {
		drawLine(w, b, p)
		f1 := func(x float64) float64 {
			return -w[2]*x*x + w[1]*x + b
		}
		drawQuadraticFunction(p, f1)
	} else {
		drawLine(w, b, p)
	}

	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter.png"); err != nil {
		panic(err)
	}
	// special thanks to
	//https://medium.com/@balazs.dianiska/generating-heatmaps-with-go-83988b22c000
}

type plottable struct {
	grid [][]float64
	N    int
	M    int
	f    func(c, r int) float64
}

func (p plottable) Dims() (c, r int) {
	return p.N, p.M
}
func (p plottable) X(c int) float64 {
	return float64(c)
}
func (p plottable) Y(r int) float64 {
	return float64(r)
}
func (p plottable) Z(c, r int) float64 {
	return p.f(c, r)
}
