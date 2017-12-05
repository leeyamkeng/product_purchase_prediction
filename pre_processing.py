# @Author: Lee Yam Keng
# @Date: 2017-10-25 3:12:16
# @Last Modified by: Lee
# @Last Modified time: 2017-12-03 17:41:58

import numpy as np
import pandas as pd

def isNaN(num):
		return num != num

def build_base_player_vec(player_set):
	"""
	stack all of the columns on top of each other so that we have a list of
	all possible player ids. Then from this we can make a set(unique ids) of
	the player ids.


	Parameters
	----------
	player_set : pandas dataframe of the starting players

	Returns
	-------
	a list that looks like ['player_index_k', ...] where k is a player id

	"""
	players = pd.DataFrame()
	for col in player_set:
		players = pd.concat([players, player_set[col]], axis=0)
	players.columns = ['player_index']
	players = pd.get_dummies(players, dummy_na=True, columns=['player_index'])
	return list(players.columns)

def encode_column(player_set):
	"""
	Convert our player ids into a useable input for our
	forecasting model. Technically this is a little different then
	normal indicators signaling the presence of a player. This is b/c
	if more than one player are unknown in the starting line up then our
	"player_index_" will contain a value greater than 1.

	Parameters
	----------
	player_set : pandas DataFrame of all the starting players.

	Returns
	-------
	a k-hot encoded matrix of starting players

	"""
	player_indices = build_base_player_vec(player_set)
	print ("player_indices", player_indices)
	exit()

	encoded_player_set = pd.DataFrame()
	for _, row in player_set.iterrows():
		indicators = pd.DataFrame([0] * len(player_indices)).T
		indicators.columns = player_indices

		# perhaps coming up with a better way to handle NaNs would be useful
		# right now I think they just show up as "player_index_"
		#
		for p in row:
			print ('------------ p -----------', p)
			if p is None:
				print ("error encoding None")
			# elif math.isnan(p):
			# 	print ("error encoding NaN")
			else:
				indicators["player_index_{}".format(p)] += 1
		encoded_player_set = pd.concat([encoded_player_set, indicators],
										axis=0)

	encoded_player_set.columns = player_indices
	return encoded_player_set

def processing():

	df = pd.read_csv('dataset/prodhit_2017-11-02.gz', compression='gzip', usecols=['mcvisid', 'visit_start_time_gmt', 'prodid', 'shopcat', 'purchase'])

	hl_cats = []
	array_shopcat = df['shopcat']
	for item in array_shopcat:
		pass
		if isNaN(item):
			pass
			hl_cats.append(item)
		else:			
			splited_items = item.split('-')

			if len(splited_items) > 1:			
				hl_cats.append(splited_items[0] + '-' + splited_items[1])
			else:
				hl_cats.append(splited_items[0])
	
	df['hl_cats'] = hl_cats

	encoded_shopcats = pd.get_dummies(df['hl_cats'])
	df = df.drop('shopcat', axis=1)
	df = df.drop('hl_cats', axis=1)
	df = df.join(encoded_shopcats)

	df.dropna(subset=['prodid'], inplace=True)

	# new_list = []
	# old_mcvisid = ''
	# old_time = ''

	# for index, row in df.iterrows():
	# 	print ('row', row)
	# 	print ('index', index)
	# 	if row['mcvisid'] == old_mcvisid and row['visit_start_time_gmt'] == old_time:
	# 		pass
	# 	else:
	# 		if index == 0:
	# 			continue
	# 		else:
	# 			new_list.append(df[index - 1])
	# 		old_mcvisid = row['mcvisid']
	# 		old_time = row['visit_start_time_gmt']

	# print ('new_list', new_list)
	# return new_list
	return df
	# print ('df', df)
	# print ('encoded_shopcats', encoded_shopcats.shape)

	# df.to_csv('encoded_file.csv', sep='\t')

if __name__ == '__main__':
	processing()
	pass