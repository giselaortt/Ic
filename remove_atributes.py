import pandas as pd

def remove_atributes( nome_atual, nome_novo ):
	data = pd.read_csv( nome_atual )
	to_be_deleted = [ 'sensor_id',
			#'exp',
			'file',
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
	data.to_csv( nome_novo, index = False)
