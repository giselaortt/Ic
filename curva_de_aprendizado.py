#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

'''
Funcao que plota a curva de aprendizado do modelo.
Espera o dataset como um numpy array e o modelo não treinado.

modelo: objeto que deve implementar os metodos fit e predict

model_name: string com o nome do modelo
	    Opcional

number_of_splits: determina em quantas vezes dividiremos o dataset. É o numeoro de pontos no grafico (tente diminuir caso esteja tomando muito tempo)

number_of_runs: numero de vezes em que o modelo sera testado( a cada vez é selecionado um novo dataset de teste. No final será incluida a media dos testes.
'''
def curva_aprendizagem( modelo, dataset, test_dataset = None, model_name = None, number_of_splits = 100, number_of_runs = 5, y_start = 90 ):
	plt.suptitle( 'curva de aprendizado para ' + model_name, fontsize = 16 )
	plt.xlabel('tamanho do dataset')
	plt.ylabel('accuracy')
	plt.grid()
	plt.ylim(y_start, 100)
	dataset_size = dataset.shape[0]
	Y = dataset[:,:1]
	X = dataset[:,1:]
	line_one = []
	line_two = []
	x_values = []
	for i in range(1, number_of_splits ):
		size = (int)( dataset_size*(i)/(number_of_splits) )
		x_values.append(size)
		accuracy_test_set = []
		accuracy_train_set = []
		for i in range( number_of_runs ):
			x_train, x_test, y_train, y_test = train_test_split( X, Y.flatten(), train_size = size, test_size = dataset.shape[0]-size, shuffle = False )
			modelo.fit( x_train, y_train.flatten() )
			if test_dataset is None:
				y_pred = modelo.predict( x_test )
				accuracy_test_set.append( accuracy_score( y_pred, y_test )*100 )
			else:
				y_pred = modelo.predict(test_dataset[:,1:])
				accuracy_test_set.append( accuracy_score( y_pred, test_dataset[:,:1]))

			y_pred = modelo.predict( x_train )
			accuracy_train_set.append( accuracy_score( y_pred, y_train )*100 )
		line_one.append( np.mean(accuracy_test_set) )
		line_two.append( np.mean(accuracy_train_set) )
		
	plt.plot( x_values, line_one, 'o-', color = "g", label = 'test set' )
	plt.plot( x_values, line_two, 'o-',color = "b", label = 'training set' )
	plt.legend( loc = "best" )
	plt.figure()
		
	return


def normalizar( dataset ):
	for col in range( 1,  dataset.shape[1] ):
		dataset[:,col] = dataset[:,col]/np.amax(dataset[:,col])
	return dataset


def running( train_input, test_input, train_output, test_output, classifier, name ):
	classifier.fit( train_input, train_output.flatten() )
	predicted = classifier.predict( test_input )
	accuracy = accuracy_score(test_output, predicted, normalize = True)
	print '\t', name, 'accuracy:', accuracy*100


def remove_atributes( dataset ):
	to_be_deleted = [ 'sensor_id',
			'exp',
			#'file',
			'time_elapsed',
			'year',
			'month',
			'day',
			'min',
			'sec',
			'humidity',
			'luminosity',
			'altitude',
			'air_pressure' ]
	for name in to_be_deleted:
		data.drop( name, axis = 1, inplace = True )

