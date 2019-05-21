from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from matplotlib import pyplot as plt

def prepare_data():
	# Mining hardware data: Extracted manually from multiple sources
	# Columns: device (string), release_date (date), hashrate (TH/s), power_usage (W/GH), price (USD)
	mining_hw = pd.read_csv('mining_hw.csv', sep=",", skipinitialspace=True, parse_dates=True, comment="#")

	# Bitcoin prices, hash rates, and transaction fees: From Blockchain.info on 2019-03-19
	btc_price = pd.read_csv('market-price.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'price'])

	btc_hashrate = pd.read_csv('hash-rate.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'hashrate'])
	btc_tx_fees = pd.read_csv('transaction-fees.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'tx_fee'])
	btc_txs = pd.read_csv('n-transactions.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'tx_count'])
	btc_difficulty = pd.read_csv('difficulty.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'difficulty'])

	mining_hw['release_date'] = mining_hw['release_date'].astype('datetime64[D]')
	btc_price['date'] = btc_price['date'].astype('datetime64[D]')
	btc_hashrate['date'] = btc_hashrate['date'].astype('datetime64[D]')
	btc_tx_fees['date'] = btc_tx_fees['date'].astype('datetime64[D]')
	btc_txs['date'] = btc_txs['date'].astype('datetime64[D]')
	btc_difficulty['date'] = btc_difficulty['date'].astype('datetime64[D]')

	# Calculate changes in hash rate
	btc_hashrate['change'] = btc_hashrate['hashrate'].diff()
	btc_hashrate['change'] = btc_hashrate['change'].fillna(0.0)
	
	# Block rewards in USD per block (halvings are hard-coded)
	halving = [pd.to_datetime('2012-11-28 00:00:00'), pd.to_datetime('2016-07-09 00:00:00')]
	conditions = [(btc_price.date < halving[0]),
	(btc_price.date < halving[1]) & (btc_price.date >= halving[0]),
	(btc_price.date >= halving[1])]
	block_rwd = [50.0, 25.0, 12.5]
	btc_price['block_reward'] = np.select(conditions, block_rwd, default=0.0)
	
	btc = pd.concat([btc_price, btc_hashrate, btc_tx_fees, btc_txs, btc_difficulty], axis=1, sort=False, join='inner')
		
	btc = btc.replace({'difficulty': 0}, 1e-12)
	btc = btc.loc[:, ~btc.columns.duplicated()]

	return mining_hw, btc

def model(btc, mining_hw):
	bulk_threshold = 5000
	ms = []
	new_hw = []
	old_hw = []
	curr_mrkt_shr = np.array([0.0 for i in mining_hw['device'].values])
	storage = np.array([0.0 for i in mining_hw['device'].values])
	for row in btc.itertuples():
		hashrate_change = ((row.dif_change * row.block_count_smooth) / (144 * 600 * 1e12 / 2**32) + row.smooth_hashrate_diff) / 2.0
		if hashrate_change > 0:
			#TODO add condition to check if it is profitable to turn on this hardware
			leftover = hashrate_change
			i = len(curr_mrkt_shr) - 1
			while leftover > 0 and i >= 0:
				if row.rev_per_hashrate / mining_hw.iloc[i]['power_usage'] < 0.04:
					break
				if storage[i] == 0:
					i -= 1
					continue
				if storage[i] - leftover >= 0:
					curr_mrkt_shr[i] += leftover
					storage[i] -= leftover
					leftover = 0
				else:
					leftover -= storage[i]
					curr_mrkt_shr[i] += storage[i]
					storage[i] = 0
					i -= 1
			
			hashrate_change = leftover
			if leftover > 0:
				hw_type = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
				curr_mrkt_shr[hw_type] += hashrate_change
				if hashrate_change / mining_hw.iloc[hw_type]['hashrate'] > bulk_threshold:
					new_hw.append([row.date, hw_type, hashrate_change])
		else:
			hashrate_change *= -1
			leftover = -hashrate_change
			i = 0
			while leftover < 0 and i < len(curr_mrkt_shr):
				if curr_mrkt_shr[i] == 0:
					i += 1
					continue
				if curr_mrkt_shr[i] + leftover >= 0:
					if hashrate_change / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						old_hw.append([row.date, i, hashrate_change])
					curr_mrkt_shr[i] += leftover
					storage[i] -= leftover
					leftover = 0
				else:
					if hashrate_change / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						old_hw.append([row.date, i, hashrate_change])
					leftover += curr_mrkt_shr[i]
					storage[i] += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		ms.append([row.date] + curr_mrkt_shr.tolist())
	market_share = pd.DataFrame(ms, columns=['date'] + mining_hw['device'].values.tolist())
	return market_share, new_hw, old_hw

def main():
	mining_hw, btc = prepare_data()
	# v Crazy inverse formula to get block count from hash rate reported by blockchain.info v
	btc['block_count'] = (144 * 600 * 1e12 / 2**32) * (btc['hashrate'] / btc['difficulty'])
	w = 3 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2, center=True).mean()
	btc['block_count_smooth'] = btc['block_count_avg'].rolling(w, min_periods=w//2, center=True).mean()
	
	btc['hashrate_smooth'] = btc['hashrate'].rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean()
	btc['smooth_hashrate_diff'] = btc['hashrate_smooth'].diff().fillna(0.0)
	
	
	btc['dif_change'] = btc['difficulty'].diff().fillna(0)
	btc['hw_change'] = btc['dif_change'] != 0
	
	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	
	market_share, new_hw, old_hw = model(btc, mining_hw)
	
	bought = pd.DataFrame(new_hw, columns=['date', 'device', 'hashrate'])
	b2 = pd.merge(btc[['date', 'rev_per_hashrate']], bought[['date', 'device']])
	b3 = pd.merge(b2[['device']], mining_hw[['power_usage']], how='left', left_on='device', right_index=True)
	b2['prof_price'] = b2['rev_per_hashrate']/b3['power_usage']
	sold = pd.DataFrame(old_hw, columns=['date', 'device', 'hashrate'])
	s2 = pd.merge(btc[['date', 'rev_per_hashrate']], sold[['date', 'device']])
	s3 = pd.merge(s2[['device']], mining_hw[['power_usage']], how='left', left_on='device', right_index=True)
	s2['prof_price'] = s2['rev_per_hashrate']/s3['power_usage']
	
	fig, ax = plt.subplots()
	ax.scatter(s2['date'].values, s2['prof_price'].values, c='r')
	ax.scatter(b2['date'].values, b2['prof_price'].values, c='g')
	plt.show()
	
	y = []
	for i in market_share.loc[:, market_share.columns != 'date'].columns.values:
		y.append(market_share[i])
	
	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	ax.set_ylim((0, 0.6e8))
	ax.stackplot(btc['date'].values, y, labels=market_share.columns[1:].values)
	ax.plot(btc['date'].values, btc['hashrate'].values, c='b', alpha=0.5)
	ax2 = ax.twinx()
	ax2.set_ylim((0, 22000))
	ax2.set_yscale('linear')
	ax2.plot(btc['date'].values, btc['price'].values, c='r')
	plt.show()
	
	print('Buying')
	print(b2['date'].count()/btc['date'].count())
	print(b2.describe())
	print('Selling')
	print(s2['date'].count()/btc['date'].count())
	print(s2.describe())

if __name__ == "__main__":
	main()
