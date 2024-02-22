package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"
)

func main() {

	//####################### Data #########################

	//filename := "data/exams1.csv"
	filename := "data/arcs.csv"

	x, y, lineMinX, lineMaxX := readData(filename) //gettings data sets

	//fmt.Println("lineMinX:", lineMinX)
	//fmt.Println("lineMaxX:", lineMaxX)

	xTrain, xTest, yTrain, yTest := split(x, y) //splitting data sets

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
