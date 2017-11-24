#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import curva_de_aprendizado as ca
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from tables import tables
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def main():
	dataset = pd.read_csv("Archive/data_44100_1024_treino.csv").as_matrix()
	dataset2 = pd.read_csv("Archive/data_44100_1024_teste.csv").as_matrix()

	mlp = MLPClassifier(hidden_layer_sizes=(60), learning_rate = 'constant', max_iter = 1000, activation = 'logistic', learning_rate_init = 0.01, shuffle = False )
	rf = RandomForestClassifier(100, max_depth = 7)
	dataset = ca.normalizar(dataset)
	dataset2 = ca.normalizar(dataset2)
	np.random.shuffle(dataset)
	np.random.shuffle(dataset2)

	plt.title("data_44100_1024_treino e dados normalizados", fontsize = 12)
	ca.curva_aprendizagem( mlp, dataset[:600], y_start = 0, model_name = "Multilayer Perceptron", number_of_splits = 200, number_of_runs=10 )
	
	dataset = np.concatenate((dataset, dataset2))

	plt.title("data_44100_1024_treino e data_44100_1024_teste com dados normalizados", fontsize = 12)
	ca.curva_aprendizagem( mlp, dataset[:600], y_start = 0, model_name = "Multilayer Perceptron", number_of_splits = 200, number_of_runs=10 )

	plt.title("data_44100_1024_treino e data_44100_1024_teste com altura limitada(7)", fontsize = 12)
	ca.curva_aprendizagem( rf, dataset, y_start = 90 ,model_name = "Random Forest", number_of_splits = 200, number_of_runs = 1 )

	plt.show()

main()
