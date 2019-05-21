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

def hashrate_share(mining_hw, btc, threshold):
	everyday_reduction = 1
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	test = []
	mrkt_shr_hist = []
	conf = []
	rmv = []
	nhw = []
	curr_mrkt_shr = np.array([0.0 for i in range(len(cols)-1)])
	idle = np.array([0.0 for i in range(len(cols)-1)])

	for row in btc.itertuples():
		rev_per_hw = curr_mrkt_shr * row.rev_per_hashrate
		profitable_prices = row.rev_per_hashrate / mining_hw['power_usage']
		nonprofitable_hws = profitable_prices < 0.01
		curr_mrkt_shr[nonprofitable_hws] = curr_mrkt_shr[nonprofitable_hws] * 0.7
		curr_mrkt_shr[curr_mrkt_shr < 1e-12] = 0
		curr_used = np.nonzero(curr_mrkt_shr)
		curr_mrkt_shr[curr_used] = curr_mrkt_shr[curr_used] * everyday_reduction
		
		#if np.abs(blocks - row.block_count_avg) > threshold:
		if row.change_in_hw:
			if 0 < row.change: # Buy new hardware
				if idle.sum() < row.change:
					curr_mrkt_shr = curr_mrkt_shr + idle
					idle = np.array([0.0 for i in range(len(cols)-1)])
				else:
					curr_mrkt_shr = curr_mrkt_shr + ((row.change / idle.sum()) * idle)
					idle = idle * (row.change / idle.sum())
				hw_type = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
				ch = row.hashrate - row.hashrate_avg
				curr_mrkt_shr[hw_type] += ch
				nhw.append([row.date, hw_type, ch])
			else: # Turn off old hardware
				old_ones = mining_hw[mining_hw['release_date'] <= row.date].index.tolist()
				leftover = row.hashrate - row.hashrate_avg
				i = 0
				while leftover < 0 and i < len(old_ones)-1:
					if curr_mrkt_shr[i] == 0:
						i += 1
						continue
					if curr_mrkt_shr[i] + leftover >= 0:
						if profitable_prices[i] < 0.08:
							idle[i] -= leftover
						else:
							rmv.append([row.date, i, -leftover])
						curr_mrkt_shr[i] += leftover
						leftover = 0
					else:
						if profitable_prices[i] < 0.08:
							idle[i] += curr_mrkt_shr[i]
						else:
							rmv.append([row.date, i, curr_mrkt_shr[i]])
						leftover += curr_mrkt_shr[i]
						curr_mrkt_shr[i] = 0
						i += 1
		tmp = curr_mrkt_shr.sum()
		blocks = 144 * 600 * (tmp * 1e12 / row.difficulty) / 2**32
		if blocks < row.block_count_avg - threshold:
			curr_mrkt_shr = curr_mrkt_shr + idle
			idle = np.array([0.0 for i in range(len(cols)-1)])
			tmp = curr_mrkt_shr.sum()
			if tmp < row.hashrate:
				hw_type = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
				curr_mrkt_shr[hw_type] += row.hashrate - tmp
		elif blocks > row.block_count_avg + threshold:
			leftover = row.hashrate - tmp
			i = 0
			while leftover < 0 and i < len(curr_mrkt_shr):
				if curr_mrkt_shr[i] == 0:
					i += 1
					continue
#				if row.rev_per_hashrate / mining_hw['power_usage'][i] > 0.2:
#					print(row.date, row.rev_per_hashrate / mining_hw['power_usage'][i])
				if curr_mrkt_shr[i] + leftover >= 0:
					idle[i] -= leftover
					curr_mrkt_shr[i] += leftover
					leftover = 0
				else:
					idle[i] += curr_mrkt_shr[i]
					leftover += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		curr_mrkt_shr[curr_mrkt_shr < 1e-12] = 0
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr.tolist())
	turned_off = pd.DataFrame(rmv, columns=['date', 'deviceID', 'hashrate'])
	confidence = pd.DataFrame(conf, columns=['date', 'conf'])
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	new_hw = pd.DataFrame(nhw, columns=['date', 'deviceID', 'hashrate'])
	return market_share, confidence, turned_off, new_hw

def main():
	mining_hw, btc = prepare_data()
	# v Crazy inverse formula to get block count from hash rate reported by blockchain.info v
	btc['block_count'] = (144 * 600 * 1e12 / 2**32) * (btc['hashrate'] / btc['difficulty'])
	w = 3 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2).mean()
	btc['hashrate_avg'] = btc['hashrate'].rolling(w, min_periods=w//2).mean()
	# threshold to consider the block count as normal fluctuations
	threshold = 20.5 # 28 for 98%, 23.5 for 95%, 20.5 for 90%
	btc['change_in_hw'] = np.abs(btc['block_count'] - btc['block_count_avg']) > threshold

	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	
	market_share, confidence, turned_off, new_hw = hashrate_share(mining_hw, btc, threshold)
	
	electricity_prices = []
	for row in turned_off.itertuples():
		power_saved = mining_hw['power_usage'][row.deviceID]
		money_not_earned = btc[btc['date'] == row.date]['rev_per_hashrate'].values[0]
		electricity_prices.append((row.date, row.deviceID, row.hashrate, money_not_earned / power_saved))
	
	w=5
	
	e_price = pd.DataFrame(electricity_prices, columns=['date', 'deviceID', 'hashrate', 'price'])
	e_price['sparsity'] = e_price['date'].diff()/np.timedelta64(1, 'D')
	e_price['sparsity'] = e_price['sparsity'].fillna(1)
	e_price['sparsity'] = e_price['sparsity'].rolling(w, min_periods=w//2, center=True).mean()
	
	x = pd.DataFrame(btc[btc['date'].isin(turned_off['date'])]['rev_per_hashrate']).T
	y = pd.DataFrame(mining_hw['power_usage'])

	z = y.values.dot(x.values)
	z[np.isnan(z)] = 0
	
	electricity_prices = []
	for row in new_hw.itertuples():
		power_used = mining_hw['power_usage'][row.deviceID]
		money_earned = btc[btc['date'] == row.date]['rev_per_hashrate'].values[0]
		electricity_prices.append((row.date, row.deviceID, row.hashrate, money_earned / power_used))
	
	ep = pd.DataFrame(electricity_prices, columns=['date', 'deviceID', 'hashrate', 'price'])
	ep['sparsity'] = ep['date'].diff()/np.timedelta64(1, 'D')
	ep['sparsity'] = ep['sparsity'].fillna(1)
	ep['sparsity'] = ep['sparsity'].rolling(w, min_periods=w//2, center=True).mean()
	
	print(e_price['date'].count() / btc['date'].count(), ep['date'].count() / btc['date'].count())
	checking = e_price['date']
	print(e_price[['date', 'deviceID', 'price']])
	print(btc[btc['date'].isin(checking.values)][['date', 'price', 'hashrate', 'rev_per_hashrate', 'change', 'block_count', 'block_count_avg']])

	checking = ep[ep['date'] > pd.to_datetime('2018-01-01 00:00:00')]['date']
	print(ep[ep['price'] > 0][['date', 'deviceID', 'price']])
	print(btc[btc['date'].isin(checking.values)][['date', 'price', 'hashrate', 'rev_per_hashrate', 'change', 'block_count', 'block_count_avg']])
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', xlim=(pd.to_datetime('2009-01-01 00:00:00'), pd.to_datetime('2019-05-10 00:00:00')), ylim=(0, 200), zlim=(0, 1))
	ax.scatter(e_price['date'].values, e_price['sparsity'].values, e_price['price'], c='r', alpha=0.5)
	ax.scatter(ep[ep['price'] < 1]['date'].values, ep[ep['price'] < 1]['sparsity'].values, ep[ep['price'] < 1]['price'], c='b', alpha=0.5)
	plt.show()

	fig, ax = plt.subplots()
	ax.hist(e_price['price'], bins=50, facecolor='r', alpha=0.5)
	ax.hist(ep['price'], bins=500, facecolor='b', alpha=0.5)
	plt.show()

	y2 = []
	for i in market_share.loc[:, market_share.columns != 'date'].columns.values:
		y2.append(market_share[i])
	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	ax.stackplot(btc['date'].values, y2, labels=market_share.columns.values[1:])
	ax.plot(btc['date'].values, btc['hashrate'])
	ax.legend(loc='upper left')
	plt.show()

if __name__ == "__main__":
	main()
