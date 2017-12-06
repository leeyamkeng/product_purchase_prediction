# @Author: Lee Yam Keng
# @Date: 2017-10-25 3:12:16
# @Last Modified by: Lee
# @Last Modified time: 2017-12-03 17:41:58

import numpy as np
import pandas as pd

def isNaN(num):
	return num != num

def processing():

	df = pd.read_csv('dataset/prodhit_2017-11-02.gz', compression='gzip', usecols=['mcvisid', 'visit_start_time_gmt', 'prodid', 'shopcat', 'purchase'])
	
	# reduce the duplicated rows
	new_df = pd.DataFrame(columns = ['mcvisid', 'visit_start_time_gmt', 'prodid', 'shopcat', 'purchase'])
	old_mcvisid = ''
	old_time = ''
	for index, row in df.iterrows():
		if row['mcvisid'] == old_mcvisid and row['visit_start_time_gmt'] == old_time:
			pass
		else:
			if index == 0:
				continue
			else:
				if pd.isnull(df.iloc[index - 1]['prodid']):
					continue
				else:
					new_df.append(df.iloc[[index - 1]])
			old_mcvisid = row['mcvisid']
			old_time = row['visit_start_time_gmt']

	# convert the shopcats into high-levels
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
	encoded_prodid = pd.get_dummies(df['prodid'])
	df = df.join(encoded_prodid)

	# print ('df[1:5000]', df[1:5000]) # [4999 rows x 12144 columns]
	# exit()

	return df

if __name__ == '__main__':
	processing()
	pass