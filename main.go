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
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	epochs              = 5e5
	printEveryNthEpochs = 1e4
	funcType            = 3 // 1 = linear, 2 = quadratic polynomial, 3 = cubic polynomial
	dataPath            = "arcs"
)

type decBoundPlot struct {
	rows, cols int
	f          func(c, r int) float64
}

func (p decBoundPlot) Dims() (c, r int)   { return p.cols, p.rows }
func (p decBoundPlot) Z(c, r int) float64 { return p.f(c, r) }
func (p decBoundPlot) X(c int) float64    { return float64(c) }
func (p decBoundPlot) Y(r int) float64    { return float64(r) }

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

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func dot(a []float64, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func p(x, w []float64, b float64) float64 {
	return sigmoid(dot(x, w) + b)
}

func inference(inputs [][]float64, w []float64, b float64) (res []float64) {
	for _, x := range inputs {
		res = append(res, p(x, w, b))
	}
	return res
}

func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64) {
	dw = make([]float64, len(polynomial(0, 0)))
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
		if y[i] == math.Round(p(x, w, b)) {
			res++
		}
	}
	return res / float64(len(y)) * 100
}

func polynomial(x1, x2 float64) (res []float64) {
	for i := 0; i <= funcType; i++ {
		for j := 0; j <= funcType-i; j++ {
			res = append(res, math.Pow(x1, float64(i))*math.Pow(x2, float64(j)))
		}
	}
	return res
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
	inputPointsMaxX, inputPointsMaxY := inputs[0][0], inputs[0][1]
	for i := range inputs {
		if inputs[i][0] > inputPointsMaxX {
			inputPointsMaxX = inputs[i][0]
		}
		if inputs[i][1] > inputPointsMaxX {
			inputPointsMaxY = inputs[i][1]
		}
	}

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
	for i := range inputs {
		inputs[i] = polynomial(inputs[i][0], inputs[i][1])
	}
	xTrain, xTest, yTrain, yTest := split(inputs, labels)
	img := make(chan *image.RGBA, 1)
	render := func(x *image.RGBA) {
		select {
		case <-img:
			img <- x
		case img <- x:
		}
	}

	go func() {
		learningRate := 1e-3
		w := make([]float64, len(polynomial(0, 0)))
		var b, squaredGradB float64
		squaredGradW := make([]float64, len(w))
		epsilon := 1e-8
		for i := 0; i <= epochs; i++ {
			y := inference(xTrain, w, b)
			dw, db := dCost(xTrain, yTrain, y)
			for i := range dw {
				squaredGradW[i] += dw[i] * dw[i]
				w[i] -= (learningRate / math.Sqrt(squaredGradW[i]+epsilon)) * dw[i]
			}
			squaredGradB += db * db
			b -= (learningRate / math.Sqrt(squaredGradB+epsilon)) * db
			if i%printEveryNthEpochs == 0 {
				boundPlot := decBoundPlot{
					rows: int(inputPointsMaxY),
					cols: int(inputPointsMaxX),
					f: func(c, r int) float64 {
						x := polynomial(float64(c), float64(r))
						return p(x, w, b)
					},
				}
				plotters := []plot.Plotter{
					plotter.NewContour(boundPlot, []float64{0.5}, palette.Heat(1, 255)),
				}
				plotters = append(plotters, inputsScatter[0])
				plotters = append(plotters, inputsScatter[1])
				legend := fmt.Sprintf("Accuracy: %.2f", accuracy(inputs, labels, w, b))
				render(Plot(legend, plotters...))
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
