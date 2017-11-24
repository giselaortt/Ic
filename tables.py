#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

'''
imprime as tabelas de acertos e erros para um modelo. Por enquanto, implementado apenas para 2 classes.
o modelo é treinado na propria função e as classes devem vir como 1 e 2.

model: deve implementar as funções fit e predict
name1: nome da classe 1
name2: nome da classe 2
'''
def tables( train_x, test_x, train_y, test_y, model, name1, name2 ):
	model.fit( train_x, train_y.flatten() )
	out = [ round(model.predict( elem.reshape(1, -1) )) for elem in test_x ]
	print 'total correct predicted', accuracy_score( out, test_y, normalize = False) 
	print 'sample size', train_x.shape[0] , "\n" 
	correct = [_==__ for _,__ in zip( out,test_y )]

	one_correct =  np.sum( [ _ == True and cl == 1 for _, cl in zip(correct,  test_y) ] ) 
	two_correct =  np.sum( [ _ == True and cl == 2 for _, cl in zip(correct,  test_y) ] )
	one_false = np.sum( [ _ == False and cl == 1 for _, cl in zip(correct,  test_y) ] )
	two_false = np.sum( [ _ == False and cl ==2 for _, cl in zip(correct,  test_y) ] )

	print " "*10 + '|' + "{:<10}".format(name1)+ '|' + "{:<10}".format(name2) + ' ===> Expected classes'
	print "{:<10}".format(name1) + '|' + "{:<10}".format(one_correct) + '|' + "{:<10}".format(two_false)
	print "{:<10}".format(name2) + '|' + "{:<10}".format(one_false) + '|' + "{:<10}".format(two_correct)
	print "^"
	print "predicted classes"

	one_correct = round( float(one_correct)/float(train_x.shape[0])*100 , 2 )
	two_correct = round( float(two_correct)/float(train_x.shape[0])*100 , 2 )
	one_false = round( float(one_false)/float(train_x.shape[0])*100 , 2 )
	two_false = round(  float(two_false)/float(train_x.shape[0])*100 , 2 )
	print ""

	print " "*10 + '|' + "{:<10}".format(name1)+ '|' + "{:<10}".format(name2) + ' ===> Expected classes'
	print "{:<10}".format(name1) + '|' + "{:<10}".format(one_correct) + '|' + "{:<10}".format(two_false)
	print "{:<10}".format(name2) + '|' + "{:<10}".format(one_false) + '|' + "{:<10}".format(two_correct)
	print "^"
	print "predicted classes"

