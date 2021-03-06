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
	
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	test = []
	mrkt_shr_hist = []
	conf = []
	rmv = []
	nhw = []
	curr_mrkt_shr = np.array([0.0 for i in range(len(cols)-1)])

	for row in btc.itertuples():
		rev_per_hw = curr_mrkt_shr * row.rev_per_hashrate
		profitable_prices = mining_hw['power_usage'] / row.rev_per_hashrate
		nonprofitable_hws = profitable_prices < 0.01
		curr_mrkt_shr[nonprofitable_hws] = curr_mrkt_shr[nonprofitable_hws] * 0.7
		curr_used = np.nonzero(curr_mrkt_shr)
		curr_mrkt_shr[curr_used] = curr_mrkt_shr[curr_used] * everyday_reduction
		tmp = curr_mrkt_shr.sum()
		
		blocks = 144 * 600 * (tmp * 1e12 / row.difficulty) / 2**32
		
		#if np.abs(blocks - row.block_count_avg) > threshold:
		if row.change_in_hw:
			if tmp < row.hashrate: # Buy new hardware
				hw_type = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
				curr_mrkt_shr[hw_type] += row.hashrate - tmp
				nhw.append([row.date, hw_type, row.hashrate - tmp])
			else: # Turn off old hardware
				old_ones = mining_hw[mining_hw['release_date'] <= row.date].index.tolist()
				leftover = row.hashrate - tmp
				i = 0
				while leftover < 0:
					if curr_mrkt_shr[i] == 0:
						i += 1
						continue
					if curr_mrkt_shr[i] + leftover >= 0:
						rmv.append([row.date, i, -leftover])
						curr_mrkt_shr[i] += leftover
						leftover = 0
					else:
						rmv.append([row.date, i, curr_mrkt_shr[i]])
						leftover += curr_mrkt_shr[i]
						curr_mrkt_shr[i] = 0
						i += 1
			if tmp == row.hashrate:
				tmp -= 0.1
			conf.append([row.date] + [100000.0/(np.abs(tmp - row.hashrate))])
		tmp = curr_mrkt_shr.sum()
		blocks = (144 * 600 * 1e12 / 2**32) * (tmp / row.difficulty)
		if np.abs(blocks - row.block_count_avg) > threshold:
			if tmp < row.hashrate: # Buy new hardware
				hw_type = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
				curr_mrkt_shr[hw_type] += row.hashrate - tmp
				nhw.append([row.date, hw_type, row.hashrate - tmp])
			else: # Turn off old hardware
				old_ones = mining_hw[mining_hw['release_date'] <= row.date].index.tolist()
				leftover = row.hashrate - tmp
				i = 0
				while leftover < 0:
					if curr_mrkt_shr[i] == 0:
						i += 1
						continue
					if curr_mrkt_shr[i] + leftover >= 0:
						rmv.append([row.date, i, -leftover])
						curr_mrkt_shr[i] += leftover
						leftover = 0
					else:
						rmv.append([row.date, i, curr_mrkt_shr[i]])
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
	w = 10 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2, center=True).mean()
	
	# threshold to consider the block count as normal fluctuations
	threshold = 20.5 # 28 for 98%, 23.5 for 95%, 20.5 for 90%
	btc['change_in_hw'] = np.abs(btc['block_count'] - btc['block_count_avg']) > threshold

	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)

	market_share, confidence, turned_off, new_hw = hashrate_share(mining_hw, btc, threshold)
	investment = market_share.loc[:, market_share.columns != 'date'].dot(mining_hw['price'].values)
	
	fig, ax = plt.subplots()
	ax.plot(btc['date'].values, investment.values)
	plt.show()

if __name__ == "__main__":
	main()
