import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from os import listdir, path
from statistics import mean, median, stdev
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

def dir_to_dataframe(directory):
	"""Concatenate all csv files in directory and returns as pandas DataFrame"""
	filepaths = [path.join(directory, f) for f in listdir(directory) if f.endswith('.csv')]
	df = pd.concat(map(pd.read_csv, filepaths))
	return df


def preprocess_data(raw_features, labels):
	"""

	Preprocesses the data by:
		1. Removing duplicate labels and joining both dataframes on bookingID
		2. Unflatten trips by making each trip one row (all data points for that column is put into a list sorted by time)
		3. Computing aggregate features from raw features using tsfresh and joining to combined dataframe
		4. Computing derivatives of certain columns and their aggregates
		5. Returning time series features and aggregate features as separate dataframes

	Warning: this takes a while (10+ mins on all the given data)
	

	Returns:
	pd.DataFrame: bookingIDs
	pd.DataFrame: labels
	pd.DataFrame: aggregate_features
	pd.DataFrame: timeseries_features

	"""

	#drop all trips with duplicate labels since there are only 26 of them
	deduplicated_labels = labels.drop_duplicates(subset = 'bookingID', keep = False).set_index(keys=['bookingID']).sort_index()
	filtered_features = raw_features[raw_features.bookingID.isin(deduplicated_labels.index)]

	#make each time series into a list in it's row
	timeseries_features = raw_features.sort_values(by='second').groupby('bookingID').agg(list).reset_index()
	combined = timeseries_features.join(deduplicated_labels, on='bookingID', how='inner')

	#tsfresh gives features which are aggregates of the time series, e.g. mean, median, standard deviation, etc
	extracted_features = extract_features(raw_features, column_id="bookingID", column_sort="second", default_fc_parameters=MinimalFCParameters())
	combined = combined.merge(extracted_features, left_on="bookingID", right_index=True, how="inner")

	#get derivatives such as jerk (second order derivative of velocity)
	combined = combined.assign(jerk_x=get_derivative_list(combined, 'acceleration_x'), jerk_y=get_derivative_list(combined, 'acceleration_y'), jerk_z=get_derivative_list(combined, 'acceleration_z'), gyro_accel_x=get_derivative_list(combined, 'gyro_x'), gyro_accel_y=get_derivative_list(combined, 'gyro_y'), gyro_accel_z=get_derivative_list(combined, 'gyro_z'), gps_accel=get_derivative_list(combined, 'Speed'))

	#compute aggregates for derivative functions
	combined['min_jerk_x'] = np.vectorize(min)(combined['jerk_x'])
	combined['min_jerk_y'] = np.vectorize(min)(combined['jerk_y'])
	combined['min_jerk_z'] = np.vectorize(min)(combined['jerk_z'])

	combined['max_jerk_x'] = np.vectorize(max)(combined['jerk_x'])
	combined['max_jerk_y'] = np.vectorize(max)(combined['jerk_y'])
	combined['max_jerk_z'] = np.vectorize(max)(combined['jerk_z'])

	combined['mean_jerk_x'] = np.vectorize(mean)(combined['jerk_x'])
	combined['mean_jerk_y'] = np.vectorize(mean)(combined['jerk_y'])
	combined['mean_jerk_z'] = np.vectorize(mean)(combined['jerk_z'])

	combined['median_jerk_x'] = np.vectorize(median)(combined['jerk_x'])
	combined['median_jerk_y'] = np.vectorize(median)(combined['jerk_y'])
	combined['median_jerk_z'] = np.vectorize(median)(combined['jerk_z'])

	combined['stdev_jerk_x'] = np.vectorize(stdev)(combined['jerk_x'])
	combined['stdev_jerk_y'] = np.vectorize(stdev)(combined['jerk_y'])
	combined['stdev_jerk_z'] = np.vectorize(stdev)(combined['jerk_z'])


	combined['min_gyro_accel_x'] = np.vectorize(min)(combined['gyro_accel_x'])
	combined['min_gyro_accel_y'] = np.vectorize(min)(combined['gyro_accel_y'])
	combined['min_gyro_accel_z'] = np.vectorize(min)(combined['gyro_accel_z'])

	combined['max_gyro_accel_x'] = np.vectorize(max)(combined['gyro_accel_x'])
	combined['max_gyro_accel_y'] = np.vectorize(max)(combined['gyro_accel_y'])
	combined['max_gyro_accel_z'] = np.vectorize(max)(combined['gyro_accel_z'])

	combined['mean_gyro_accel_x'] = np.vectorize(mean)(combined['gyro_accel_x'])
	combined['mean_gyro_accel_y'] = np.vectorize(mean)(combined['gyro_accel_y'])
	combined['mean_gyro_accel_z'] = np.vectorize(mean)(combined['gyro_accel_z'])

	combined['median_gyro_accel_x'] = np.vectorize(median)(combined['gyro_accel_x'])
	combined['median_gyro_accel_y'] = np.vectorize(median)(combined['gyro_accel_y'])
	combined['median_gyro_accel_z'] = np.vectorize(median)(combined['gyro_accel_z'])

	combined['stdev_gyro_accel_x'] = np.vectorize(stdev)(combined['gyro_accel_x'])
	combined['stdev_gyro_accel_y'] = np.vectorize(stdev)(combined['gyro_accel_y'])
	combined['stdev_gyro_accel_z'] = np.vectorize(stdev)(combined['gyro_accel_z'])


	combined['min_gps_accel'] = np.vectorize(min)(combined['gps_accel'])

	combined['max_gps_accel'] = np.vectorize(max)(combined['gps_accel'])

	combined['mean_gps_accel'] = np.vectorize(mean)(combined['gps_accel'])

	combined['median_gps_accel'] = np.vectorize(median)(combined['gps_accel'])

	combined['stdev_gps_accel'] = np.vectorize(stdev)(combined['gps_accel'])


	#prepare output
	booking_id = combined['bookingID']
	labels = combined['label']
	aggregate_features_columns = [
		'min_jerk_x', 
		'min_jerk_y', 
		'min_jerk_z', 
		'max_jerk_x', 
		'max_jerk_y', 
		'max_jerk_z', 
		'mean_jerk_x', 
		'mean_jerk_y', 
		'mean_jerk_z', 
		'median_jerk_x', 
		'median_jerk_y', 
		'median_jerk_z', 
		'stdev_jerk_x', 
		'stdev_jerk_y', 
		'stdev_jerk_z', 
		'min_gyro_accel_x', 
		'min_gyro_accel_y', 
		'min_gyro_accel_z', 
		'max_gyro_accel_x', 
		'max_gyro_accel_y', 
		'max_gyro_accel_z', 
		'mean_gyro_accel_x', 
		'mean_gyro_accel_y', 
		'mean_gyro_accel_z', 
		'median_gyro_accel_x', 
		'median_gyro_accel_y', 
		'median_gyro_accel_z', 
		'stdev_gyro_accel_x', 
		'stdev_gyro_accel_y', 
		'stdev_gyro_accel_z', 
		'min_gps_accel', 
		'max_gps_accel', 
		'mean_gps_accel', 
		'median_gps_accel', 
		'stdev_gps_accel'
		]

	final_timeseries_features = combined.drop(['bookingID', 'label', 'second'] + list(extracted_features.columns) + aggregate_features_columns, axis=1)
	aggregate_features = combined[list(extracted_features.columns) + aggregate_features_columns]

	return booking_id, labels, aggregate_features, final_timeseries_features

def get_derivative_list(df, column_name):
	"""Returns list of derivatives applied to each row in a column, where each row value is a list"""
	return [list(np.gradient(r)) for r in [np.asarray(r) for r in df[column_name]]]



def make_timeseries_same_length(timeseries_features, max_length = 520):
	"""
	Makes all input timeseries the same length by truncating those longer than max_length and padding those shorter than max_length

	Default max_length is 520 as that is length of the median trip in seconds

	Returns:
	np.array: timeseries
	"""
	 # median length of trip
	timeseries = np.zeros((timeseries_features.shape[0], timeseries_features.shape[1], max_length))
	for i, column in enumerate(timeseries_features.columns):
		seq = list(timeseries_features[column])
		padded_sequence = pad_sequences(seq, maxlen=max_length, dtype='float64', padding='post', truncating='post')
		scaler = StandardScaler()
		padded_sequence = scaler.fit_transform(padded_sequence)
		timeseries[:,i,:] = padded_sequence

	return timeseries
