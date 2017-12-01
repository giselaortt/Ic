import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt


"""
            model: should come trained
            dataset: numpy objct. shouldnt contain the classes.
"""
def counting( dataset, window_size, model ):
        x = []
        y = []
        for i in range( dataset.shape[0] - window_size ) :
            temp = model.predict_proba( dataset[i:i+window_size, 1:] )
            print( temp )
            return


def main():
        data = np.loadtxt( "Archive/data_44100_1024_treino.csv", skiprows = 1, delimiter = ',')
        test = np.loadtxt("Archive/data_44100_1024_teste.csv", skiprows = 1, delimiter = ',' )
        data = data[data[:,1].argsort()]
        mlp = MLPClassifier(hidden_layer_sizes=(30),learning_rate='constant',max_iter=1000,activation='logistic',learning_rate_init=0.01,shuffle=False)
        rf = RandomForestClassifier( 100 )
        rf.fit( teste[:500,1:], teste[:500,:1] )
        mlp.fit( teste[:500,1:] , teste[:500,:1] )
        #counting( data, 100, mlp )


main()
