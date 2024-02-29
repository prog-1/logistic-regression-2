package main

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
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

func (a *App) updatePlot(w []float64, b float64, xTrain, xTest [][]float64, yTrain, yTest []float64, maxX0, maxX1 float64, f func([]float64) []float64) {

	//################# Initia1zation ##########################

	p := plot.New() //initia1zing plot

	//###################### Legend #############################

	p.Legend.Add(fmt.Sprint("Accuracy: ", accuracy(xTest, yTest, w, b)))

	//####################### Heat map ##############################

	p.Add(heatMap(maxX0, maxX1, w, b, f))

	//###################### Points #############################

	//####### Train points #######

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

	//####### Test points #######

	//tp - true prediction | fp - false prediction

	//Colors
	tpColor := color.RGBA{0, 255, 0, 255} //Dark green
	fpColor := color.RGBA{255, 0, 0, 255} //Red

	//Plotters
	var tpPlotter plotter.XYs
	var fpPlotter plotter.XYs

	predictions := inference(xTest, w, b) // getting predictions of Test points

	//Distributing Test points depending on whether prediction was correct or not
	for i, p := range predictions { //for every point

		//fmt.Println("xTest[i]:", xTest[i])
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

	//##################### Ebiten #############################

	a.plot = p //replacing old plot with new one

}

//############################# HEAT MAP ##################################

// structure for heat map
type decBoundPlot struct {
	rows, cols int
	f          func(c, r int) float64
}

func (p decBoundPlot) Dims() (c, r int)   { return p.cols, p.rows }
func (p decBoundPlot) X(c int) float64    { return float64(c) }
func (p decBoundPlot) Y(r int) float64    { return float64(r) }
func (p decBoundPlot) Z(c, r int) float64 { return p.f(c, r) } //height of each cell

func heatMap(maxX0, maxX1 float64, w []float64, b float64, f func([]float64) []float64) *plotter.HeatMap {

	boundPlot := decBoundPlot{
		rows: int(math.Ceil(maxX1) + 1),
		cols: int(math.Ceil(maxX0) + 1),
		f:    func(c, r int) float64 { return p(f([]float64{float64(c), float64(r)}), w, b) },
	}
	//return plotter.NewContour(boundPlot, []float64{0.5}, palette.Heat(1, 255))
	return plotter.NewHeatMap(boundPlot, moreland.SmoothPurpleOrange().Palette(255))
}

//###############################################################
