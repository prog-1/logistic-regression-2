package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

const (
	power = 3
)

func main() {

	//####################### Data #########################

	//filename := "data/arcs.csv"
	//filename := "data/blobs.csv"
	//filename := "data/circle.csv"
	filename := "data/exams.csv"
	//filename := "data/two_circles.csv"

	x, y, maxX0, maxX1 := readData(filename) //gettings data sets

	convertedX := convert2Ds(x) //x1, x2 => x1^2, x2^2, x1, x2

	xTrain, xTest, yTrain, yTest := split(convertedX, y) //splitting data sets

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(sW, sH)
	ebiten.SetWindowTitle("Linear Regression")

	//App instance
	a := NewApp(sW, sH)

	//####################### Logistic Regression #########################

	go func() { //Starting logistic regression in another thread
		a.regression(xTrain, xTest, yTrain, yTest, maxX0, maxX1)
	}()

	//####################### Ebiten #########################

	//Running game
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

}
