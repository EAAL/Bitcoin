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

def hashrate_share(mining_hw, btc):
	cols = ['date'] + mining_hw['device'].values.tolist()
	
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	mrkt_shr_hist = []
	rmv = []
	nhw = []
	curr_mrkt_shr = np.array([0.0 for i in range(len(cols)-1)])

	for row in btc.itertuples():
		tmp = curr_mrkt_shr.sum()
		
		if tmp < row.avg_hashrate:
			new_model = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
			curr_mrkt_shr[new_model] += row.avg_hashrate - tmp
			if tmp > 0 and row.avg_hashrate / tmp > 1.01:
				nhw.append([row.date, new_model, row.avg_hashrate - tmp])
		else:
			leftover = row.avg_hashrate - tmp
			i = 0
			while leftover < 0:
				if curr_mrkt_shr[i] == 0:
					i += 1
					continue
				if curr_mrkt_shr[i] + leftover >= 0:
					if tmp > 0 and row.avg_hashrate / tmp < 0.99:
						rmv.append([row.date, i, -leftover])
					curr_mrkt_shr[i] += leftover
					leftover = 0
				else:
					if tmp > 0 and row.avg_hashrate / tmp < 0.99:
						rmv.append([row.date, i, curr_mrkt_shr[i]])
					leftover += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		curr_mrkt_shr[curr_mrkt_shr < 1e-12] = 0
		mrkt_shr_hist.append([row.date] + curr_mrkt_shr.tolist())
	turned_off = pd.DataFrame(rmv, columns=['date', 'deviceID', 'hashrate'])
	market_share = pd.DataFrame(mrkt_shr_hist, columns=cols)
	new_hw = pd.DataFrame(nhw, columns=['date', 'deviceID', 'hashrate'])
	return market_share, turned_off, new_hw

def main():
	mining_hw, btc = prepare_data()
	# v Crazy inverse formula to get block count from hash rate reported by blockchain.info v
	btc['block_count'] = (144 * 600 * 1e12 / 2**32) * (btc['hashrate'] / btc['difficulty'])
	w = 14 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2, center=True).mean()
	btc['avg_hashrate'] = btc['hashrate'].rolling(w, min_periods=w//2, center=True).mean()
	
	btc['change_avg_hashrate'] = btc['avg_hashrate'].diff().fillna(0)

	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)

	market_share, turned_off, new_hw = hashrate_share(mining_hw, btc)
	
	electricity_prices = []
	for row in turned_off.itertuples():
		power_saved = mining_hw['power_usage'][row.deviceID]
		money_not_earned = btc[btc['date'] == row.date]['rev_per_hashrate'].values[0]
		electricity_prices.append((row.date, row.deviceID, row.hashrate, money_not_earned / power_saved))
	
	sold = pd.DataFrame(electricity_prices, columns=['date', 'deviceID', 'hashrate', 'price'])
	
	electricity_prices = []
	for row in new_hw.itertuples():
		power_used = mining_hw['power_usage'][row.deviceID]
		money_earned = btc[btc['date'] == row.date]['rev_per_hashrate'].values[0]
		electricity_prices.append((row.date, row.deviceID, row.hashrate, money_earned / power_used))
	
	bought = pd.DataFrame(electricity_prices, columns=['date', 'deviceID', 'hashrate', 'price'])
	
	y = []
	for i in market_share.columns.values[1:]:
		y.append(market_share[i])
	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	ax.stackplot(btc['date'].values, y, labels=mining_hw['device'].values)
	#ax.plot(btc['date'].values, btc['price'].values)
	#ax2 = ax.twinx()
	ax.scatter(sold[sold['price'] > 0.3]['date'].values, sold[sold['price'] > 0.3]['hashrate'].values, s=10*sold[sold['price'] > 0.3]['price'].values, c='r')
	ax.scatter(bought[bought['price'] < 0.15]['date'].values, bought[bought['price'] < 0.15]['hashrate'].values, s=10*bought[bought['price'] < 0.15]['price'].values, c='g')
	plt.show()


if __name__ == "__main__":
	main()
