package main

import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func quadraticInputX(inputs [][]float64, w []float64) (xs [][]float64, ws []float64) {
	xs, ws = inputs, w
	a, b, c := len(inputs), len(inputs[0]), len(w)
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			xs[i] = append(xs[i], xs[i][j]*xs[i][j])
		}
	}
	for i := 0; i < c; i++ {
		ws = append(ws, ws[i]*ws[i])
	}
	return xs, ws
}
func drawLine(w []float64, b float64, p *plot.Plot) {
	line, _ := plotter.NewLine(plotter.XYs{
		{X: 0, Y: (-w[0]*0 - b) / w[1]},
		{X: 100, Y: (-w[0]*100 - b) / w[1]}})
	line.Color = color.RGBA{B: 255, A: 255}
	p.Add(line)
}
func drawPolinomialFunction(p *plot.Plot, f func(x float64) float64) {
	for i := 0; i < 10; i++ {
		line, _ := plotter.NewLine(plotter.XYs{
			{X: float64(i), Y: f(float64(i))},
			{X: float64(i + 1), Y: f(float64(i + 1))}})
		line.Color = color.RGBA{B: 255, A: 255}
		p.Add(line)
	}
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
	adress := "data/blobs.csv"
	data := ReadData(adress)
	// variables
	p := plot.New()
	var dw []float64
	var db, scale float64
	xTrain, xTest, kapusta, yTrain, yTest := Split(data)
	w := make([]float64, len(xTrain[0]))
	for i := range w {
		w[i] = rand.Float64() * 2
	}
	b := rand.Float64() * 2
	alpha := 1e-3
	epochs := 1000000
	// Output formatting
	fmt.Printf("Start values of weights and bias: %v, %v: \n", w, b)
	xTrain, w = quadraticInputX(xTrain, w)
	w, b, dw, db = GradientDescent(xTrain, yTrain, w, alpha, b, epochs)
	fmt.Printf("End values of weights and bias: %v, %v: \n", w, b)
	fmt.Printf("End values of dw and db: %v, %v: \n", dw, db)
	fmt.Printf("Epochs: %v\n", epochs)
	xTest, _ = quadraticInputX(xTest, []float64{0, 0, 0, 0})
	score := Accuracy(xTest, yTest, w, b)
	fmt.Printf("Score: %v\n", score)
	// drawing
	var plotData Plottable
	plotData.grid = kapusta
	plotData.N = len(kapusta)
	plotData.M = len(kapusta)
	plotData.f = func(c, r int) float64 {
		return Sigmoid(Dot([]float64{float64(c), float64(r), float64(c * c), float64(r * r)}, w) + b)
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
		drawPolinomialFunction(p, f1)
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
