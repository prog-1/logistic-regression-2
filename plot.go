package main

import (
	"fmt"
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

// Converting plot to ebiten.Image
func PlotToImage(p *plot.Plot) *ebiten.Image {

	img := image.NewRGBA(image.Rect(0, 0, sW, sH)) //creating image.RGBA to store the plot

	c := vgimg.NewWith(vgimg.UseImage(img)) //creating plot drawer for the image

	p.Draw(draw.New(c)) //drawing plot on the image

	return ebiten.NewImageFromImage(c.Image()) //converting image.RGBA to ebiten.Image (doing in Draw)
	///Black screen issue: was giving "img" instead of "c.Image()" in the function.
}

//###################################################################################

func (a *App) updatePlot(w []float64, b float64, xTrain, xTest [][]float64, yTrain, yTest []float64, lineMinX, lineMaxX float64) {

	//################# Initialization ##########################

	p := plot.New() //initializing plot

	//#################### Legend ##############################

	p.Legend.Add(fmt.Sprint("Accuracy: ", accuracy(xTest, yTest, w, b)))

	//#################### Testing Points ##############################

	/*

		xTest
		yTest
		inference

	*/

	//#################### Train points ##############################

	//Colors
	trainColor := color.RGBA{0, 0, 0, 255} //Black

	//Plotters
	var trainPlotter plotter.XYs

	//Saving all Train points to one plotter
	for i := 0; i < len(xTrain); i++ { //for every point
		trainPlotter = append(trainPlotter, plotter.XY{X: xTrain[i][0], Y: xTrain[i][1]}) //save the point in training plotter
	}

	//Train scatter
	trueTrainScatter, _ := plotter.NewScatter(trainPlotter) //creating new scatter from point data
	trueTrainScatter.Color = trainColor
	p.Add(trueTrainScatter)

	//#################### Test points ##############################

	//tp - true prediction | fp - false prediction

	//Colors
	tpColor := color.RGBA{0, 50, 0, 255}  //Dark green
	fpColor := color.RGBA{255, 0, 0, 255} //Red

	//Plotters
	var tpPlotter plotter.XYs
	var fpPlotter plotter.XYs

	predictions := inference(xTest, w, b) // getting predictions of Test points

	//Distributing Test points depending on whether prediction was correct or not
	for i, p := range predictions { //for every point
		if p >= 0.5 { //prediction is 1
			if yTest[i] == 1 { // truth is 1
				tpPlotter = append(tpPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //prediction is correct
			} else { // truth is 0
				fpPlotter = append(fpPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //prediction is incorrect
			}
		} else { //prediction is 0
			if yTest[i] == 1 { // truth is 1
				fpPlotter = append(fpPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //prediction is incorrect
			} else { // truth is 0
				tpPlotter = append(tpPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //prediction is correct
			}
		}
	}

	//True prediction scatter
	tpScatter, _ := plotter.NewScatter(tpPlotter) //creating new scatter from point data
	tpScatter.Color = tpColor
	tpScatter.GlyphStyle.Shape = draw.PlusGlyph{} //plus form
	p.Add(tpScatter)

	//False prediction scatter
	fpScatter, _ := plotter.NewScatter(fpPlotter) //creating new scatter from point data
	fpScatter.Color = fpColor
	fpScatter.GlyphStyle.Shape = draw.PlusGlyph{} //plus form
	p.Add(fpScatter)

	//####################### Line ##############################

	/*
		(x1 = X | x2 = Y)

		0 = k*x + b
		0 = w1x1 + w2x2 + b
		-w2x2 = w1x1 + b
		w2x2 = - b - w1x1
		x2 = (-w1x1-b) / w2
	*/

	linePlotter := plotter.XYs{
		{X: lineMinX, Y: (-w[0]*lineMinX - b) / w[1]},
		{X: lineMaxX, Y: (-w[0]*lineMaxX - b) / w[1]},
	}

	line, _ := plotter.NewLine(linePlotter) //creating line

	p.Add(line) // adding line to the plot

	//##################### Ebiten #############################

	a.plot = p //replacing old plot with new one

}
