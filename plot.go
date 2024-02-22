package main

import (
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

	//#################### Points ##############################

	//Colors | true: y=1, false: y=0
	trueTrainColor := color.RGBA{0, 0, 0, 255}   //Black
	falseTrainColor := color.RGBA{0, 0, 0, 255}  //Black
	trueTestColor := color.RGBA{0, 100, 0, 255}  //Green
	falseTestColor := color.RGBA{100, 0, 0, 255} //Red

	//Plotters
	var trueTrainPlotter plotter.XYs
	var falseTrainPlotter plotter.XYs

	var trueTestPlotter plotter.XYs
	var falseTestPlotter plotter.XYs

	//Distributing the points to separate Train plotters
	for i := 0; i < len(xTrain); i++ { //for every point
		if yTrain[i] == 0 { //if the current point is false/0/negative
			trueTrainPlotter = append(trueTrainPlotter, plotter.XY{X: xTrain[i][0], Y: xTrain[i][1]}) //Saving the point in false plotter
		} else { //if the current point is true/1/positive
			falseTrainPlotter = append(falseTrainPlotter, plotter.XY{X: xTrain[i][0], Y: xTrain[i][1]}) //Saving the point in true plotter
		}
	}

	//Distributing the points to separate Test plotters
	for i := 0; i < len(xTest); i++ { //for every point
		if yTest[i] == 0 { //if the current point is false/0/negative
			trueTestPlotter = append(trueTestPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //Saving the point in false plotter
		} else { //if the current point is true/1/positive
			falseTestPlotter = append(falseTestPlotter, plotter.XY{X: xTest[i][0], Y: xTest[i][1]}) //Saving the point in true plotter
		}
	}

	//True Train scatter
	trueTrainScatter, _ := plotter.NewScatter(trueTrainPlotter) //creating new scatter from point data
	trueTrainScatter.Color = trueTrainColor
	p.Add(trueTrainScatter)

	//False Train scatter
	falseTrainScatter, _ := plotter.NewScatter(falseTrainPlotter)
	falseTrainScatter.Color = falseTrainColor
	p.Add(falseTrainScatter)

	//True Test scatter
	trueTestScatter, _ := plotter.NewScatter(trueTestPlotter)
	trueTestScatter.Color = trueTestColor
	trueTestScatter.GlyphStyle.Shape = draw.PlusGlyph{} //plus form
	p.Add(trueTestScatter)

	//False Test scatter
	falseTestScatter, _ := plotter.NewScatter(falseTestPlotter)
	falseTestScatter.Color = falseTestColor
	falseTestScatter.GlyphStyle.Shape = draw.PlusGlyph{} //plus form
	p.Add(falseTestScatter)

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
