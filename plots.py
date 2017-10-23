import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

first =  pd.read_csv( "dataset_Bees_FruitFly-1kxTemp.csv")
second = pd.read_csv("dataset_Bees_Colmeia_USP.csv")

# we want correlation btween humidity, luminosity, altitude and air_pressure and other variables
fruitfly = first[ [ cl == 1 for cl in first['class'] ] ]

print np.argmax( fruitfly.luminosity )

'''plt.scatter( fruitfly.wbf, fruitfly.humidity )
plt.ylabel("wbf")
plt.xlabel('humidity')
plt.show()'''

plt.scatter( second.wbf, second.luminosity )
plt.ylabel("wbf")
plt.xlabel('luminosity' )
#plt.xticks( range( np.argmax( fruitfly.luminosity ) ) )
#plt.yticks( range( np.argmax( fruitfly.wbf )))
plt.show()

'''plt.scatter( fruitfly.wbf, fruitfly.altitude )
plt.ylabel("wbf")
plt.xlabel('altitude')
plt.show()'''

'''plt.scatter( fruitfly.wbf, fruitfly.air_pressure )
plt.ylabel("wbf")
plt.xlabel('air_pressure')
plt.show()'''
