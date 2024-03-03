package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	screenWidth, screenHeight        = 640, 480
	epochs                           = 100000
	learningRateW                    = 1e-4
	learningRateB                    = 1e-1
	functionPower                    = 2
	inputPointsMinX, inputPointsMaxX = 0, 100
	printEveryXEpoch                 = 100
	scale                            = 10
)

type decBoundPlot struct {
	rows, cols int
	f          func(c, r int) float64
}

func (p decBoundPlot) Dims() (c, r int)   { return p.cols * scale, p.rows * scale }
func (p decBoundPlot) Z(c, r int) float64 { return p.f(int(float64(c)/scale), int(float64(r)/scale)) }
func (p decBoundPlot) X(c int) float64    { return float64(c) / scale }
func (p decBoundPlot) Y(r int) float64    { return float64(r) / scale }

func Plot(legend string, ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	p.Legend.Add(legend)
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	file, err := os.Open("data/circle.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	inputs := make([][]float64, len(data))
	for i := range inputs {
		inputs[i] = make([]float64, 2)
	}
	y := make([]float64, len(data))
	for i, row := range data {
		for j := 0; j < 2; j++ {
			inputs[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		y[i], _ = strconv.ParseFloat(row[2], 64)
	}
	w := make([]float64, len(polynomial(0, 0)))
	for i := range w {
		w[i] = rand.Float64()
	}

	inputPointsMaxX, inputPointsMaxY := inputs[0][0], inputs[0][1]
	for i := range inputs {
		if inputs[i][0] > inputPointsMaxX {
			inputPointsMaxX = inputs[i][0]
		}
		if inputs[i][1] > inputPointsMaxY {
			inputPointsMaxY = inputs[i][1]
		}
	}

	xTrain, xTest, yTrain, yTest := split(inputs, y)
	b := 0.2

	//
	xys := make([]plotter.XYs, 2)
	for i := range inputs {
		if y[i] == 0 {
			xys[0] = append(xys[0], plotter.XY{X: inputs[i][0], Y: inputs[i][1]})
		} else {
			xys[1] = append(xys[1], plotter.XY{X: inputs[i][0], Y: inputs[i][1]})
		}
	}
	var inputsScatter []*plotter.Scatter
	for i := range xys {
		tmp, _ := plotter.NewScatter(xys[i])
		inputsScatter = append(inputsScatter, tmp)
	}
	inputsScatter[0].Color = color.RGBA{255, 0, 0, 255}
	inputsScatter[1].Color = color.RGBA{0, 255, 0, 255}
	for i := range inputs {
		inputs[i] = polynomial(inputs[i][0], inputs[i][1])
	}
	img := make(chan *image.RGBA, 1)
	render := func(x *image.RGBA) {
		select {
		case <-img:
			img <- x
		case img <- x:
		}
	}
	//

	for i := 0; i <= epochs; i++ {
		fxi := inference(xTrain, w, b)
		dw, db := deratives(xTrain, yTrain, fxi)
		cost := cost(len(xTrain), fxi, yTrain)
		for j := 0; j < len(w); j++ {
			w[j] = w[j] - learningRateW*dw[j]
		}
		b = b - learningRateB*db
		if i%printEveryXEpoch == 0 {
			fmt.Printf("Epoch number: %d\ndw: %f\ndb: %f\ncost: %f\n", i, dw, db, cost)
		}
	}
	wTrained, bTrained := w, b
	//

	boundPlot := decBoundPlot{
		rows: int(inputPointsMaxY + 1.5),
		cols: int(inputPointsMaxX + 1.5),
		f: func(c, r int) float64 {
			x := polynomial(float64(c), float64(r))
			return inferenceForOne(x, wTrained, bTrained)
		},
	}

	plotters := []plot.Plotter{
		plotter.NewContour(boundPlot, []float64{0.5}, palette.Heat(1, 255)),
		// plotter.NewHeatMap(boundPlot, palette.Heat(5, 255)),
	}

	plotters = append(plotters, inputsScatter[0])
	plotters = append(plotters, inputsScatter[1])
	legend := fmt.Sprintf("Accuracy: %.2f", accuracy(xTest, yTest, wTrained, bTrained))
	render(Plot(legend, plotters...))
	// fmt.Println("Final accuracy:", accuracy(xTest, yTest, wTrained, bTrained))
	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}

func deratives(inputs [][]float64, y, fxi []float64) (dw []float64, db float64) {
	dw = make([]float64, len(polynomial(0, 0)))
	n := len(inputs)
	for i := 0; i < n; i++ {
		for j := 0; j < len(inputs[0]); j++ {
			dw[j] += ((fxi[i] - y[i]) * inputs[i][j]) / float64(n)
		}
		db += (fxi[i] - y[i]) / float64(n)
	}
	return dw, db
}

func inference(inputs [][]float64, w []float64, b float64) []float64 {
	var predictions []float64
	for _, x := range inputs {
		predictions = append(predictions, linearModel(x, w, b))
	}
	return predictions
}

func sigmoid(z float64) float64 { return 1 / (1 + math.Pow(math.E, -z)) }

func linearModel(x, w []float64, b float64) (res float64) { return sigmoid(dot(x, w) + b) }

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func loss(p, y float64) float64 {
	if y == 1 {
		return -math.Log(p)
	} else {
		return -math.Log(1 - p)
	}
}

func cost(n int, fxi, y []float64) (res float64) {
	for i := 0; i < n; i++ {
		l := loss(fxi[i], y[i])
		// if math.IsInf(l,1){
		// 	fmt.Println(fxi[i], y[i])
		// }
		res += l
	}
	return res / float64(n)
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	trainIndices := make(map[int]bool)
	for i := 0; i < len(inputs)/5*4; i++ {
		idx := rand.Intn(len(inputs))
		for trainIndices[idx] {
			idx = rand.Intn(len(inputs))
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

func accuracy(inputs [][]float64, y []float64, w []float64, b float64) float64 {
	var res float64
	for i, x := range inputs {
		if y[i] == math.Round(inferenceForOne(x, w, b)) {
			res++
		}
	}
	return res / float64(len(y)) * 100
}

func polynomial(x1, x2 float64) (res []float64) {
	for i := 0; i <= functionPower; i++ {
		for j := 0; j <= functionPower-i; j++ {
			res = append(res, math.Pow(x1, float64(i))*math.Pow(x2, float64(j)))
		}
	}
	return res
}

func inferenceForOne(x, w []float64, b float64) float64 {
	return sigmoid(dot(x, w) + b)
}
