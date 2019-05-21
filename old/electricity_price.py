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
	btc_difficulty = pd.read_csv('difficulty.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'difficulty'])

	mining_hw['release_date'] = mining_hw['release_date'].astype('datetime64[D]')
	btc_price['date'] = btc_price['date'].astype('datetime64[D]')
	btc_hashrate['date'] = btc_hashrate['date'].astype('datetime64[D]')
	btc_tx_fees['date'] = btc_tx_fees['date'].astype('datetime64[D]')
	btc_txs['date'] = btc_txs['date'].astype('datetime64[D]')
	btc_difficulty['date'] = btc_difficulty['date'].astype('datetime64[D]')

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
	(btc_price['date'] >= halving[1])]
	block_rwd = [50.0, 25.0, 12.5]
	btc_price['block_reward'] = np.select(conditions, block_rwd, default=0.0)
	
	return mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs, btc_difficulty


def electricity_price(mining_hw, btc_hashrate, btc_price, btc_tx_fees):
	
	btc_rev = pd.DataFrame([], columns=['date', 'revenue'])
	btc_rev['date'] = btc_price['date']
	btc_rev['revenue'] = btc_price['price']*(6.0*btc_price['block_reward'] + btc_tx_fees['fee']/24.0)
	
	rev_per_hashrate = btc_rev['revenue'] / btc_hashrate['smooth']

	power_usage_inv = pd.DataFrame([(1.0/mining_hw['power_usage']).values.tolist() for i in btc_rev['date']])
	
	e_price = power_usage_inv.mul(rev_per_hashrate, axis=0)

	return e_price

def main():
	mining_hw, btc_price, btc_hashrate, btc_tx_fees, btc_txs, btc_difficulty = prepare_data()
	# Max electricity price profitable for each hardware
	e_price = electricity_price(mining_hw, btc_hashrate, btc_price, btc_tx_fees)
	
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.plot(btc_price['date'], e_price)
	ax.axhline(y=0.01)
	ax.axhline(y=0.04)
	ax.axhline(y=0.08)
	ax.axhline(y=0.12)
	plt.show()

if __name__ == "__main__":
	main()
