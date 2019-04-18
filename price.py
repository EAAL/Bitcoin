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
	btc_tx_fees = pd.read_csv('transaction-fees.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'fee'])
	btc_txs = pd.read_csv('n-transactions.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'n'])
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
	conditions = [(btc_price['date'] < halving[0]),
	(btc_price['date'] < halving[1]) & (btc_price['date'] >= halving[0]),
	(btc_price['date'] >= halving[1])]
	block_rwd = [50.0, 25.0, 12.5]
	btc_price['block_reward'] = np.select(conditions, block_rwd, default=0.0)
	
	return mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs, btc_difficulty

def hashrate_share(mining_hw, btc_hashrate, btc_rev, mu, std):
	
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	rev_per_hashrate = btc_rev['revenue'] / btc_hashrate['hashrate']
	test = []
	mrkt_shr_hist = []
	conf = []
	rmv = []
	curr_mrkt_shr = np.array([0.0 for i in range(len(cols)-1)])
	for row in btc_hashrate.itertuples():
		rev_per_hw = curr_mrkt_shr * rev_per_hashrate[row.Index]
		curr_used = np.nonzero(curr_mrkt_shr)
		tmp = curr_mrkt_shr.sum()
		if np.abs(tmp - row.hashrate) > std:
			if tmp < row.hashrate: # Buy new hardware
				curr_mrkt_shr[mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]] += row.hashrate - tmp
			else: # Turn off old hardware
				old_ones = mining_hw[mining_hw['release_date'] <= row.date].index.tolist()
				leftover = row.hashrate - tmp
				i = 0
				while leftover < 0:
					if curr_mrkt_shr[i] + leftover >= 0:
						rmv.append([row.date, i, leftover])
						curr_mrkt_shr[i] += leftover
						leftover = 0
					else:
						rmv.append([row.date, i, curr_mrkt_shr[i]])
						leftover += curr_mrkt_shr[i]
						curr_mrkt_shr[i] = 0
						i += 1
		conf.append([row.date] + [100000.0/(np.abs(tmp - row.hashrate)+1e-100)])
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr.tolist())
	turn_off = pd.DataFrame(rmv, columns=['date', 'deviceID', 'hashrate'])
	confidence = pd.DataFrame(conf, columns=['date', 'conf'])
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	return market_share, confidence, turn_off

def main():
	mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs, btc_difficulty = prepare_data()
	btc_blocks = pd.DataFrame([], columns=['date', 'count', 'avg'])
	btc_blocks['date'] = btc_hashrate['date']
	btc_blocks['count'] = 144 * 600 * (btc_hashrate['hashrate'] * 1e12 / btc_difficulty['difficulty']) / 2**32
	btc_blocks['per_hour'] = btc_blocks['count'] / 24
	btc_blocks['avg'] = btc_blocks['count'].rolling(20, min_periods=20//2, center=True).mean()
	std = 40
	btc_blocks['change'] = np.abs(btc_blocks['count'] - btc_blocks['avg']) > std
	print(btc_blocks[btc_blocks['change']]['change'].count())
	

	# Revenue in USD per hour
	btc_rev = pd.DataFrame([], columns=['date', 'revenue'])
	btc_rev['date'] = btc_price['date']
	btc_rev['revenue'] = btc_price['price']*(6.0*btc_price['block_reward'] + btc_tx_fees['fee']/24.0)

	market_share, confidence, turn_off = hashrate_share(mining_hw, btc_hashrate, btc_rev, mu, std)

	# Cost in USD per hour
	btc_cost = pd.DataFrame([], columns=['date', 'cost'])
	btc_cost['date'] = market_share['date']
	btc_cost['cost'] = (market_share.loc[:, market_share.columns != 'date'].dot(mining_hw['power_usage'].values))*electricity_price

	cost_share = (market_share.loc[:, market_share.columns != 'date']*mining_hw['power_usage'].values)*electricity_price
	rev_share = (market_share.loc[:, market_share.columns != 'date'].div(btc_hashrate['hashrate'], axis=0)).mul(btc_rev['revenue'], axis=0)
	

	plt.show()

if __name__ == "__main__":
	main()
