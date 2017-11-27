#import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier


'''
	model: should come trained
	dataset: numpy objct. shouldnt contain the classes.
'''
def counting( dataset, window_size, model ):
	x = []
	y = []
	for i in range( dataset.shape[0] - window_size ) :
		model.predict( dataset[i:i+window_size] )
		y.append(  )

	return


def main():
	data = np.loadtxt( "Ic/Archive/data_44100_1024_treino.csv", skiprows = 1, delimiter = ',')
	test = np.loadtxt("Ic/Archive/data_44100_1024_teste.csv", skiprows = 1, delimiter = ',' )
	data = data[data[:,1].argsort()]
	
	

main()
