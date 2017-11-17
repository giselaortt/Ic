#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import remove_atributes as ra 
import curva_de_aprendizado as ca
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from tables import tables
from remove_atributes import remove_atributes

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def test(dataset, classifier):
	x = dataset[:,1:]
	y  = dataset[:,0:1]
	train_input, test_input, train_output, test_output = train_test_split(x, y, test_size=0.2, random_state=42)
	classifier.fit( train_input, train_output.flatten() )
	predicted = classifier.predict( test_input )
	accuracy = accuracy_score(test_output, predicted, normalize = True)
	print '\t','accuracy:', accuracy*100
	

def main():
	#remove_atributes( "Archive/dataset_Bees_FruitFly-1kxTemp.csv", "Archive/first_dataset.csv" )
	mlp = MLPClassifier(hidden_layer_sizes=(30), learning_rate = 'constant', activation = 'logistic', learning_rate_init = 0.01, shuffle = False )
	rf = RandomForestClassifier(n_estimators = 100)
	dataset = pd.read_csv("Archive/data_44100_1024_treino.csv")
	#first = pd.read_csv("Archive/first_dataset.csv")
	dataset = dataset.as_matrix()
	#first = first.as_matrix()
	np.random.shuffle( dataset )
	test( dataset, mlp )
	#ca.curva_aprendizagem( mlp, dataset, "multilayer perceptron no primeiro dataset", number_of_splits = 200, number_of_runs = 1 )
	#ca.curva_aprendizagem( mlp, first, "multilayer perceptron no primeiro dataset", number_of_splits = 200, number_of_runs = 1 )

main()
