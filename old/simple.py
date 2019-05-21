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

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def main():
	mining_hw, btc = prepare_data()
	# v Crazy inverse formula to get block count from hash rate reported by blockchain.info v
	btc['block_count'] = (144 * 600 * 1e12 / 2**32) * (btc['hashrate'] / btc['difficulty'])
	w = 7 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2, center=True).mean()
	btc['prev_block_avg'] = btc['block_count_avg'].shift(1)
	
	# threshold to consider the block count as normal fluctuations, based on a Poisson process with lambda=144
	threshold = 20.5 # 28 for 98%, 23.5 for 95%, 20.5 for 90%
	
	btc['change_in_hw'] = np.abs(btc['block_count'] - btc['block_count_avg']) > threshold

	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	
	print(len(btc[btc['change_in_hw'] & (btc['date'] > pd.to_datetime('2001-07-01'))]['date'].values) / len(btc[btc['date'] > pd.to_datetime('2001-07-01')]['date'].values))
	
	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	ax.plot(btc['date'].values, btc['block_count'].values)
	ax2 = ax.twinx()
	ax2.plot(btc['date'].values, btc['difficulty'].values, c='g')
	ax.scatter(btc[btc['change_in_hw']]['date'].values, btc[btc['change_in_hw']]['block_count'].values, c='r')
	plt.show()

if __name__ == "__main__":
	main()
