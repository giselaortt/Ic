#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import curva_de_aprendizado as ca
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import tables


'''
Dataset should be an numpy array.
'''
def normalizar( dataset ):
	dataset = dataset.astype(float)
	for i in range( dataset.shape[1] ):
		dataset[:,i] = dataset[:,i] / np.amax( dataset[:,i])
	return dataset


def gerar_curva_de_aprendizado():
	treino = pd.read_csv("Archive/data_44100_1024_treino.csv").as_matrix()
	teste = pd.read_csv("Archive/data_44100_1024_teste.csv").as_matrix()

	mlp = MLPClassifier(hidden_layer_sizes=(60), learning_rate = 'constant', max_iter = 1000, activation = 'logistic', learning_rate_init = 0.01, shuffle = False )
	rf = RandomForestClassifier(100, max_depth = 7)
	treino = ca.normalizar(treino)
	teste = ca.normalizar(teste)
	np.random.shuffle(treino)
	np.random.shuffle(teste)

	plt.title("data_44100_1024_treino e dados normalizados", fontsize = 12)
	ca.curva_aprendizagem( mlp, treino[:600], y_start = 0, model_name = "Multilayer Perceptron", number_of_splits = 200, number_of_runs=10 )
	
	treino = np.concatenate((treino, teste))

	plt.title("data_44100_1024_treino e data_44100_1024_teste com dados normalizados", fontsize = 12)
	ca.curva_aprendizagem( mlp, treino[:600], y_start = 0, model_name = "Multilayer Perceptron", number_of_splits = 200, number_of_runs=10 )

	plt.title("data_44100_1024_treino e data_44100_1024_teste com altura limitada(7)", fontsize = 12)
	ca.curva_aprendizagem( rf, treino, y_start = 90 ,model_name = "Random Forest", number_of_splits = 200, number_of_runs = 1 )

	plt.show()


def gerar_tabela_de_confusao():
	tabela = tables.ConfusionTable( "fruitfly", "jatai" )
	treino = pd.read_csv("Archive/data_44100_1024_treino.csv").as_matrix()
#	treino = normalizar( treino )
	np.random.shuffle(treino)
	mlp = MLPClassifier(hidden_layer_sizes=(60), learning_rate = 'constant', max_iter = 1000, activation = 'logistic', learning_rate_init = 0.01, shuffle = False )
	teste = pd.read_csv("Archive/data_44100_1024_teste.csv").as_matrix()
#	teste = normalizar(teste)
	tabela.gerar( model = mlp, train_y = treino[:,0], test_y = teste[:,0], train_x = treino[:,1:], test_x = treino[:,1:])
	tabela.imprimir()
	tabela.imprimir_porcentagem()


gerar_curva_de_aprendizado()
