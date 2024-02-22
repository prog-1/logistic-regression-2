package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	epochs                           = 1e4
	printEveryNthEpochs              = 500
	learningRateW                    = 1e-3
	learningRateB                    = 0.5
	inputPointsMinX, inputPointsMaxX = 0, 3e2
	funcType                         = 1 // 1 = linear, 2 = quadratic polynomial, 3 = cubic polynomial
	dataPath                         = "arcs"
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

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	for _, x := range inputs {
		res = append(res, sigmoid(dot(x, w)+b))
	}
	return res
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, funcType*2)
	m := float64(len(inputs))
	for i := range inputs {
		diff := p[i] - y[i]
		for j, x := range inputs[i] {
			dw[j] += x * diff / m
		}
		db += diff / m
	}
	return
}

func split(inputs [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	xTrain, xTest = inputs[:len(inputs)*8/10], inputs[len(inputs)*8/10:]
	yTrain, yTest = y[:len(y)*8/10], y[len(y)*8/10:]
	return
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

func main() {
	file, err := os.Open("data/" + dataPath + ".csv")
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
	var labels []float64
	for i, row := range data {
		for j, col := range row {
			if j == 0 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				inputs[i] = append(inputs[i], v)
			} else if j == 1 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				inputs[i] = append(inputs[i], v)
			} else if j == 2 {
				v, err := strconv.ParseFloat(col, 64)
				if err != nil {
					log.Fatal(err)
				}
				labels = append(labels, v)
			}
		}
	}
	for i := 1; i < funcType; i++ {
		for j := range inputs {
			inputs[j] = append(inputs[j], math.Pow(inputs[j][0], funcType))
			inputs[j] = append(inputs[j], math.Pow(inputs[j][1], funcType))
		}
	}
	xTrain, xTest, yTrain, yTest := split(inputs, labels)

	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Logistic Regression")

	xys := make([]plotter.XYs, 2)
	for i := range inputs {
		if labels[i] == 0 {
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

	go func() {
		w := make([]float64, funcType*2)
		var b float64
		for i := 0; i <= epochs; i++ {
			y := inference(xTrain, w, b)
			dw, db := dCost(xTrain, yTrain, y)
			for i := range dw {
				w[i] -= dw[i] * learningRateW
			}
			b -= db * learningRateB
			if i%printEveryNthEpochs == 0 {
				xs := []float64{inputPointsMinX, inputPointsMaxX}
				resLine, _ := plotter.NewLine(plotter.XYs{{X: xs[0], Y: -(w[0]*xs[0] + b) / w[1]}, {X: xs[1], Y: -(w[0]*xs[1] + b) / w[1]}})
				render(Plot(inputsScatter[0], inputsScatter[1], resLine))
				fmt.Printf(`Epoch #%d
                dw: %.4f, db: %.4f
                w: %.4f, b: %.4f
                accuracy: %.2f
                `, i, dw, db, w, b, accuracy(xTrain, yTrain, w, b))
			}
		}
		fmt.Printf("Accuracy: %.2f", accuracy(xTest, yTest, w, b))
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}
