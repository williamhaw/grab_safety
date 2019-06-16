import pandas as pd
from os import listdir, path

def dir_to_dataframe(directory):
	"""Concatenate all csv files in directory and returns as pandas DataFrame"""
	filepaths = [path.join(directory, f) for f in listdir(directory) if f.endswith('.csv')]
	df = pd.concat(map(pd.read_csv, filepaths))
	return df

