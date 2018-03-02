#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


'''
	imprime as tabelas de acertos e erros para um modelo. Por enquanto, implementado apenas para 2 classes.
	o modelo é treinado na propria função e as classes devem vir como 1 e 2.
	model: deve implementar as funções fit e predict
	self.name1: nome da classe 1
	self.name2: nome da classe 2
	only two classes by now
'''

class ConfusionTable:
	def __init__(self, name1, name2):
		self.name1  = name1
		self.name2 = name2
		self.total = None


	def imprimir(self):
		print(" "*10 + '|' + "{:<10}".format(self.name1)+ '|' + "{:<10}".format(self.name2) + ' ===> Expected classes' )
		print("{:<10}".format(self.name1) + '|' + "{:<10}".format(self.table[0][0]) + '|' + "{:<10}".format(self.table[0][1]))
		print("{:<10}".format(self.name2) + '|' + "{:<10}".format(self.table[1][0]) + '|' + "{:<10}".format(self.table[1][1]))
		print("^")
		print("predicted classes")


	def imprimir_porcentagem(self):
		percentage_table = (self.table /float(self.total)*100).round(2)
		print("")
		print(" "*10 + '|' + "{:<10}".format(self.name1)+ '|' + "{:<10}".format(self.name2) + ' ===> Expected classes')
		print("{:<10}".format(self.name1) + '|' + "{:<10}".format(percentage_table[0][0]) + '|' + "{:<10}".format(percentage_table[0][1]))
		print("{:<10}".format(self.name2) + '|' + "{:<10}".format(percentage_table[1][0]) + '|' + "{:<10}".format(percentage_table[1][1]))
		print("^")
		print("predicted classes")


	def gerar(self, train_x, test_x, train_y, test_y, model):
		model.fit( train_x, train_y.flatten() )
		out = np.array([ model.predict( elem.reshape(1, -1))  for elem in test_x ]).round().astype(int)
		self.total = test_y.shape[0]
		correct = [ bool(_ == __) for _,__ in zip( out,test_y )]
		one_correct =  np.sum([ _ == True and cl == 1 for _, cl in zip(correct,  test_y)]) 
		two_correct =  np.sum([ _ == True and cl == 2 for _, cl in zip(correct,  test_y)])
		one_false = np.sum([ _ == False and cl == 1 for _, cl in zip(correct,  test_y)])
		two_false = np.sum([ _ == False and cl == 2 for _, cl in zip(correct,  test_y)])
		self.table = np.array([[one_correct, one_false],[two_correct,two_false]])


