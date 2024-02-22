package main

import (
	"encoding/csv"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

const (
	paramCount = 2
	seed       = 10 // for random
)

// Reads CSV file and saves data in [][] of inputs and [] of whether point is true or not
func readData(filename string) (x [][]float64, y []float64, lineMinX, lineMaxX float64) {

	// ############## Opening CSV file ##############

	file, err := os.Open(filename)

	if err != nil {
		log.Fatal(err)
	}

	defer file.Close() //Declaring file closure at the end

	// ############## Making new reader ##############

	reader := csv.NewReader(file)           // new reader
	reader.FieldsPerRecord = paramCount + 1 // dunno why it's required rlly
	reader.Comma = ','

	// ############## Reading data ##############

	records, err := reader.ReadAll() //reading all data into [][]string

	if err != nil {
		log.Fatal(err)
	}

	// ############## Declaration ##############

	x = make([][]float64, len(records))
	y = make([]float64, len(records))

	lineMinX = math.MaxFloat64 //otherwise will always be 0

	// ############## Saving data ##############

	for i, record := range records { //for each record | i is row

		// ##### Distributing record data along variables #####

		x[i] = make([]float64, len(record)) //making slice instance for every record

		for j := 0; j < len(record); j++ { //for every param besides the last one in the row | j is param

			x[i][j], err = strconv.ParseFloat(record[j], 64) //saving param in x[][]
			if err != nil {
				log.Fatal(err)
			}

		}

		// saving value in y
		y[i], err = strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			log.Fatal(err)
		}

		// updating min & max point x coordinate for decision boundary (line) starting & ending point
		if x[i][0] < lineMinX {
			lineMinX = x[i][0]
		}
		if x[i][0] > lineMaxX {
			lineMaxX = x[i][0]
		}

	}

	return x, y, lineMinX, lineMaxX
}

// Splits the x data set into the Training and Test data sets
func split(x [][]float64, y []float64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {

	rnd := rand.New(rand.NewSource(seed))

	trainIndices := make(map[int]bool)

	for i := 0; i < len(x)/5*4; i++ {
		idx := rnd.Intn(len(x))
		for trainIndices[idx] {
			idx = rnd.Intn(len(x))
		}
		trainIndices[idx] = true
	}

	for i := 0; i < len(x); i++ {
		if trainIndices[i] {
			xTrain = append(xTrain, x[i])
			yTrain = append(yTrain, y[i])
		} else {
			xTest = append(xTest, x[i])
			yTest = append(yTest, y[i])
		}
	}

	return xTrain, xTest, yTrain, yTest
}
