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
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	screenWidth, screenHeight        = 640, 480
	epochs                           = 1000
	learningRateW                    = 1e-4
	learningRateB                    = 1e-1
	functionPower                    = 1
	inputPointsMinX, inputPointsMaxX = 0, 100
)

func Plot(ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenWidth)
	file, err := os.Open("data/exams.csv")
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
	y := make([]float64, 100)
	for i, row := range data {
		for j := 0; j < 2; j++ {
			inputs[i][j], _ = strconv.ParseFloat(row[j], 64)
		}
		y[i], _ = strconv.ParseFloat(row[2], 64)
	}
	w := make([]float64, len(inputs[0]))
	for i := range w {
		w[i] = rand.Float64()
	}
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
	img := make(chan *image.RGBA, 1)
	render := func(x *image.RGBA) {
		select {
		case <-img:
			img <- x
		case img <- x:
		}
	}
	//
	xTrain, xTest, yTrain, yTest := split(inputs, y)
	b := 0.

	for i := 0; i <= epochs; i++ {
		fxi := inference(xTrain, w, b)
		dw, db := deratives(xTrain, yTrain, fxi)
		cost := cost(len(xTrain), fxi, yTrain)
		for j := 0; j < len(w); j++ {
			w[j] = w[j] - learningRateW*dw[j]
		}
		b = b - learningRateB*db
		if i%100 == 0 {
			xs := []float64{inputPointsMinX, inputPointsMaxX}
			resLine, _ := plotter.NewLine(plotter.XYs{{X: xs[0], Y: -(w[0]*xs[0] + b) / w[1]}, {X: xs[1], Y: -(w[0]*xs[1] + b) / w[1]}})
			render(Plot(inputsScatter[0], inputsScatter[1], resLine))
			fmt.Printf("Epoch nuber: %d\ndw: %f\ndb: %f\ncost: %f\n", i, dw, db, cost)
		}
	}
	wTrained, bTrained := w, b
	//

	fmt.Println("Final accuracy:", accuracy(xTest, yTest, wTrained, bTrained))
	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}

func deratives(inputs [][]float64, y, fxi []float64) (dw []float64, db float64) {
	dw = make([]float64, len(inputs[0]))
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
		if y[i] == math.Round(sigmoid(dot(x, w)+b)) {
			res++
		}
	}
	return res / float64(len(y)) * 100
}
