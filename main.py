import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def prepare_data(hash_rate_smooting_window=10):
	# Mining hardware data: Extracted manually from multiple sources
	# Columns: device (string), release_date (date), hashrate (TH/s), power_usage (W/GH), price (USD)
	mining_hw = pd.read_csv('mining_hw.csv', sep=",", skipinitialspace=True, parse_dates=True, comment="#")

	# Bitcoin prices, hash rates, and transaction fees: From Blockchain.info on 2019-03-19
	btc_price = pd.read_csv('market-price.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'price'])
	btc_hashrate = pd.read_csv('hash-rate.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'hashrate'])
	btc_tx_fees = pd.read_csv('transaction-fees.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'fee'])
	btc_txs = pd.read_csv('n-transactions.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'n'])

	mining_hw['release_date'] = mining_hw['release_date'].astype('datetime64[D]')
	btc_price['date'] = btc_price['date'].astype('datetime64[D]')
	btc_hashrate['date'] = btc_hashrate['date'].astype('datetime64[D]')
	btc_tx_fees['date'] = btc_tx_fees['date'].astype('datetime64[D]')
	btc_txs['date'] = btc_txs['date'].astype('datetime64[D]')

	# Smoothen the hashrate
	w = hash_rate_smooting_window # window size
	btc_hashrate['smooth'] = btc_hashrate['hashrate'].rolling(w, min_periods=w//2, center=True).mean()
	# Calculate changes in hash rate
	btc_hashrate['change'] = btc_hashrate['smooth'].diff()
	btc_hashrate['change'] = btc_hashrate['change'].fillna(0.0)
	
	# Block rewards in USD per block (halvings are hard-coded)
	halving = [pd.to_datetime('2012-11-28 00:00:00'), pd.to_datetime('2016-07-09 00:00:00')]
	conditions = [(btc_price['date'] < halving[0]),
	(btc_price['date'] < halving[1]) & (btc_price['date'] >= halving[0]),
	(btc_price['date'] > halving[1])]
	block_rwd = [50.0, 25.0, 12.5]
	btc_price['block_reward'] = np.select(conditions, block_rwd, default=0.0)
	
	return mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs


# Modelling the HW market

def market_share_0(mining_hw, btc_hashrate):
	# Model 0: Instant switch to new hardware
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	mrkt_shr_hist = []
	for row in btc_hashrate.itertuples():
		curr_mrkt_shr = [0.0 for i in range(len(cols)-1)]
		curr_mrkt_shr[mining_hw[mining_hw['release_date'] < row.date].tail(1).index.tolist()[0]] = row.hashrate
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr)
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	return market_share

def market_share_1(mining_hw, btc_hashrate):
	# Model 1: Assuming the new hardware is bought in increasing hash rates and the oldest gets removed in the decreases
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	mrkt_shr_hist = []
	curr_mrkt_shr = [0.0 for i in range(len(cols)-1)]
	for row in btc_hashrate.itertuples():
		if row.change >= 0:
			curr_mrkt_shr[mining_hw[mining_hw['release_date'] < row.date].tail(1).index.tolist()[0]] += row.change
		else:
			old_ones = mining_hw[mining_hw['release_date'] < row.date].index.tolist()
			leftover = row.change
			i = 0
			while leftover < 0:
				if curr_mrkt_shr[i] + leftover >= 0:
					curr_mrkt_shr[i] += leftover
					leftover = 0
				else:
					leftover += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr)
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	return market_share

def market_share_2(mining_hw, btc_hashrate):
	# Model 2: Assuming the 3 newest hardware are added in increasing hash rates and the oldest gets removed in the decreases. Maybe some people turn off their HW but then decide to use them again
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	mrkt_shr_hist = []
	curr_mrkt_shr = [0.0 for i in range(len(cols)-1)]
	for row in btc_hashrate.itertuples():
		if row.change >= 0:
			n = 3 # Get the newest n HW
			newest = mining_hw[mining_hw['release_date'] < row.date].tail(n).index.tolist()
			for i in range(len(newest)-1):
				curr_mrkt_shr[newest[(-i-1)]] += row.change * (1.0/(2.0**(i+1)))
			curr_mrkt_shr[newest[0]] += row.change * (1.0/(2.0**(len(newest)-1)))
		else:
			old_ones = mining_hw[mining_hw['release_date'] < row.date].index.tolist()
			leftover = row.change
			i = 0
			while leftover < 0:
				if curr_mrkt_shr[i] + leftover >= 0:
					curr_mrkt_shr[i] += leftover
					leftover = 0
				else:
					leftover += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr)
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	return market_share

def main():
	
	mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs = prepare_data()
	
	market_share = market_share_2(mining_hw, btc_hashrate)
	
	# Revenue in USD per day
	btc_rev = pd.DataFrame([], columns=['date', 'revenue'])
	btc_rev['date'] = btc_price['date']
	btc_rev['revenue'] = btc_price['price']*(24*6*btc_price['block_reward'] + btc_tx_fees['fee']*btc_txs['n'])

	# Cost in USD per day
	electricity_price = 0.04 # Price in USD/KWh
	btc_cost = pd.DataFrame([], columns=['date', 'cost'])
	btc_cost['date'] = market_share['date']
	btc_cost['cost'] = (market_share.loc[:, market_share.columns != 'date'].dot(mining_hw['power_usage'].values))*3600*24*electricity_price

	cost_share = (market_share.loc[:, market_share.columns != 'date']*mining_hw['power_usage'].values)*3600*24*electricity_price

	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	y = []
	for i in cost_share.columns.values:
		y.append(cost_share[i])
	ax.stackplot(btc_cost['date'].values, y, labels=cost_share.columns.values)
	ax.plot(btc_cost['date'], btc_cost['cost'], color='r')
	ax.plot(btc_rev['date'], btc_rev['revenue'], color='g')
	ax.legend(loc='upper left')
	plt.show()

if __name__ == "__main__":
	main()
