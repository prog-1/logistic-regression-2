package main

import (
	"fmt"
	"image/color"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var (
	funcType = 5
	alpha    = 1e-3
	epochs   = 1000000
)

func polynomial(x1, x2 float64) (res []float64) {
	for i := 0; i <= funcType; i++ {
		for j := 0; j <= funcType-i; j++ {
			res = append(res, math.Pow(x1, float64(i))*math.Pow(x2, float64(j)))
		}
	}
	return res
}

func drawScatter(xTrain [][]float64, yTrain []float64, p *plot.Plot) (*plotter.Scatter, *plotter.Scatter) {
	var drawR, drawG plotter.XYs
	for i := 0; i < len(yTrain); i++ {
		if yTrain[i] == 0 {
			drawR = append(drawR, struct{ X, Y float64 }{X: xTrain[i][0], Y: xTrain[i][1]})
		} else {
			drawG = append(drawG, struct{ X, Y float64 }{X: xTrain[i][0], Y: xTrain[i][1]})
		}
	}
	scatterR, err := plotter.NewScatter(drawR)
	if err != nil {
		panic(err)
	}
	scatterR.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
	scatterR.GlyphStyle.Radius = vg.Points(4)

	scatterG, err := plotter.NewScatter(drawG)
	if err != nil {
		panic(err)
	}
	scatterG.GlyphStyle.Color = color.RGBA{G: 255, A: 255}
	scatterG.GlyphStyle.Radius = vg.Points(4)

	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	return scatterR, scatterG
}
func main() {
	//reading
	adress := "data/arcs.csv"
	data := ReadData(adress)
	// variables
	p := plot.New()
	var dw []float64
	var db, b float64
	xTrain, xTest, kapusta, yTrain, yTest := Split(data)
	scatterR, scatterG := drawScatter(xTrain, yTrain, p)
	w := make([]float64, len(polynomial(0, 0)))
	geta := make([]float64, len(w))
	getab := 0.0
	// Output formatting
	fmt.Printf("Start values of weights and bias: %v, %v: \n", w, b)
	for i := range xTrain {
		xTrain[i] = polynomial(xTrain[i][0], xTrain[i][1])
	}
	w, b, dw, db = GradientDescent(xTrain, yTrain, w, geta, getab, alpha, b, epochs)
	fmt.Printf("End values of weights and bias: %v, %v: \n", w, b)
	fmt.Printf("End values of dw and db: %v, %v: \n", dw, db)
	fmt.Printf("Epochs: %v\n", epochs)
	for i := range xTest {
		xTest[i] = polynomial(xTest[i][0], xTest[i][1])
	}
	score := Accuracy(xTest, yTest, w, b)
	fmt.Printf("Score: %v\n", score)
	// drawing
	var maxX1, maxX2 float64
	for i := range kapusta {
		maxX1 = max(maxX1, kapusta[i][0])
		maxX2 = max(maxX2, kapusta[i][1])
	}
	var plotData Plottable
	plotData.N = int(maxX1) + 1
	plotData.M = int(maxX2) + 1

	plotData.f = func(c, r int) float64 {
		return Sigmoid(Dot(polynomial(float64(c), float64(r)), w) + b)
	} // w1 * r + w2 * c + b + w3 * r^2 + w4 * c^2 + w5 * r * c
	pal := moreland.ExtendedKindlmann().Palette(255)
	heatmap := plotter.NewHeatMap(plotData, pal)
	countour := plotter.NewContour(plotData, []float64{0.5}, pal)
	p.Add(heatmap)
	p.Add(countour)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "heatmap_arcs.png"); err != nil {
		panic(err)
	}
	p.Add(scatterR)
	p.Add(scatterG)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter_arcs.png"); err != nil {
		panic(err)
	}

	p.Title.Text = "LOGistic regression"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "scatter.png"); err != nil {
		panic(err)
	}
}
