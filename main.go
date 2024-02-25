package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

func main() {

	//####################### Data #########################

	filename := "data/arcs.csv"
	//filename := "data/blobs.csv"
	//filename := "data/circle.csv"
	//filename := "data/exams.csv"
	//filename := "data/two_circles.csv"

	x, y, lineMinX, lineMaxX := readData(filename) //gettings data sets

	newX := quadratic(x) //x1, x2 => x1^2, x2^2, x1, x2
	//newX := cubic(x)     //x1, x2 => x1^3, x2^3, x1^2, x2^2, x1, x2

	xTrain, xTest, yTrain, yTest := split(newX, y) //splitting data sets

	//####################### Ebiten #########################

	//Window
	ebiten.SetWindowSize(sW, sH)
	ebiten.SetWindowTitle("Linear Regression")

	//App instance
	a := NewApp(sW, sH)

	//####################### Logistic Regression #########################

	go func() { //Starting logistic regression in another thread
		a.regression(xTrain, xTest, yTrain, yTest, lineMinX, lineMaxX)
	}()

	//####################### Ebiten #########################

	//Running game
	if err := ebiten.RunGame(a); err != nil {
		log.Fatal(err)
	}

}
