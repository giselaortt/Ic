#import pandas as pd
import numpy as np


def counting( train_dataset, dataset, window_size, model  ):
	model.fit(train_dataset)
	for i in range( dataset.shape[0] - window_size ) :
		

	return


def main():
	data = np.loadtxt( "Ic/Archive/data_44100_1024_treino.csv", skiprows = 1, delimiter = ',')
	test = np.loadtxt("Ic/Archive/data_44100_1024_teste.csv", skiprows = 1, delimiter = ',' )
	data = data[data[:,1].argsort()]
	test = test[data[:,1].argsort()]


main()
