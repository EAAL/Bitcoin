from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from matplotlib import pyplot as plt

def prepare_data():
	# Mining hardware data: Extracted manually from multiple sources
	# Columns: device (string), release_date (date), hashrate (TH/s), power_usage (W/GH), price (USD)
	mining_hw = pd.read_csv('mining_hw.csv', sep=",", skipinitialspace=True, parse_dates=True, comment="#")

	# Bitcoin prices, hash rates, and transaction fees
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

	# Delivery delay for Antminers
	mining_hw.loc['S1':]['release_date'] = mining_hw.loc['S1':]['release_date'] + pd.Timedelta('30 days')
	
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
	
	# Bitcoin cash prices, hash rates
	bch_price = pd.read_csv('bch-price.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'price'])

	bch_hashrate = pd.read_csv('bch-hash-rate.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'hashrate'])
	bch_difficulty = pd.read_csv('bch-difficulty.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'difficulty'])

	mining_hw['release_date'] = mining_hw['release_date'].astype('datetime64[D]')
	bch_price['date'] = bch_price['date'].astype('datetime64[D]')
	bch_hashrate['date'] = bch_hashrate['date'].astype('datetime64[D]')
	bch_difficulty['date'] = btc_difficulty['date'].astype('datetime64[D]')

	# Calculate changes in hash rate
	bch_hashrate['change'] = bch_hashrate['hashrate'].diff()
	bch_hashrate['change'] = bch_hashrate['change'].fillna(0.0)
	
	# Block rewards in USD per block (halvings are hard-coded)
	halving = [pd.to_datetime('2012-11-28 00:00:00'), pd.to_datetime('2016-07-09 00:00:00')]
	conditions = [(bch_price.date < halving[0]),
	(bch_price.date < halving[1]) & (bch_price.date >= halving[0]),
	(bch_price.date >= halving[1])]
	block_rwd = [50.0, 25.0, 12.5]
	bch_price['block_reward'] = np.select(conditions, block_rwd, default=0.0)
	
	bch = pd.concat([bch_price, bch_hashrate, bch_difficulty], axis=1, sort=False, join='inner')
		
	bch = bch.replace({'difficulty': 0}, 1e-12)
	bch = bch.loc[:, ~bch.columns.duplicated()]

	h = np.array(mining_hw['hashrate'].tolist())
	e = np.array(mining_hw['power_usage'].tolist())
	price = np.array(mining_hw['price'].tolist()) / h

	N = len(mining_hw.index)

	break_even = []
	for i in range(1, N):
		tmp = []
		for j in range(i):
			d = int(price[i]*h[j]/((e[j]-e[i])*24))
			tmp.append(d)
			#print(d/0.05, end=" ")
		#print()
		break_even.append(tmp)
	

	return mining_hw, btc, bch, break_even

def model(btc, mining_hw, break_even, btc_to_bch, bch_to_btc):
	bulk_threshold = 1000
	ms = []
	new_hw = []
	old_hw = []
	on_again = []
	curr_mrkt_shr = np.array([0.0 for i in mining_hw['device'].values])
	storage = np.array([0.0 for i in mining_hw['device'].values])
	avg_e_price = np.array([0.0 for i in mining_hw['device'].values])
	for row in btc.itertuples():
		#hashrate_change = ((row.dif_change * row.block_count_smooth) / (144 * 600 * 1e12 / 2**32) + row.smooth_hashrate_diff) / 2.0
		hashrate_change = row.smooth_hashrate_diff
		if hashrate_change > 0:
			leftover = hashrate_change
			newest = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
			i = newest-1
			while leftover > 0 and i >= 0:

				if storage[i] == 0:
					i -= 1
					continue
				
				elec_price = avg_e_price[i]
					
				if (newest > 0) and (elec_price > 0) and ((np.abs(break_even[newest-1][i] / elec_price) < 180) or (row.rev_per_hashrate / mining_hw.iloc[i]['power_usage'] > elec_price)):
					break
				if storage[i] - leftover >= 0:
					if leftover / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						on_again.append([row.date, i, leftover])
					curr_mrkt_shr[i] += leftover
					storage[i] -= leftover
					leftover = 0
				else:
					if storage[i] / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						on_again.append([row.date, i, storage[i]])
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
			leftover = hashrate_change
			newest = mining_hw[mining_hw['release_date'] <= row.date].tail(1).index.tolist()[0]
			i = 0
			while leftover < 0 and i < len(curr_mrkt_shr):
				if curr_mrkt_shr[i] == 0:
					i += 1
					continue
				
				elec_price = row.rev_per_hashrate / mining_hw.iloc[i]['power_usage']
				
				if curr_mrkt_shr[i] + leftover >= 0:
					if (-leftover) / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						old_hw.append([row.date, i, -leftover])
					curr_mrkt_shr[i] += leftover
					if not(row.date in btc_to_bch['date'].values):
						avg_e_price[i] = (avg_e_price[i]*storage[i] + elec_price*(-leftover))/(storage[i]-leftover)
					storage[i] -= leftover
					leftover = 0
				else:
					if curr_mrkt_shr[i] / mining_hw.iloc[i]['hashrate'] > bulk_threshold:
						old_hw.append([row.date, i, curr_mrkt_shr[i]])
					leftover += curr_mrkt_shr[i]
					if not(row.date in btc_to_bch['date'].values):
						avg_e_price[i] = (avg_e_price[i]*storage[i] + elec_price*curr_mrkt_shr[i])/(storage[i]+curr_mrkt_shr[i])
					storage[i] += curr_mrkt_shr[i]
					curr_mrkt_shr[i] = 0
					i += 1
		ms.append([row.date] + curr_mrkt_shr.tolist())
	market_share = pd.DataFrame(ms, columns=['date'] + mining_hw['device'].values.tolist())
	return market_share, new_hw, old_hw, on_again, avg_e_price

def main():
	mining_hw, btc, bch, break_even = prepare_data()

	# v Crazy inverse formula to get block count from hash rate reported by blockchain.info v
	btc['block_count'] = (144 * 600 * 1e12 / 2**32) * (btc['hashrate'] / btc['difficulty'])
	w = 5 # window for averaging
	btc['block_count_avg'] = btc['block_count'].rolling(w, min_periods=w//2, center=True).mean()
	btc['block_count_smooth'] = btc['block_count_avg'].rolling(w, min_periods=w//2, center=True).mean()
	
	btc['hashrate_smooth'] = btc['hashrate'].rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean().rolling(w, min_periods=w//2, center=True).mean()
	btc['smooth_hashrate_diff'] = btc['hashrate_smooth'].diff().fillna(0.0)
	btc['change_30'] = btc['price'].diff().fillna(0.0).rolling(30).sum()
#	print(btc[(btc['change_30'] < 0) & (btc['date'] < pd.to_datetime('2012-11-28')) & (btc['date'] > pd.to_datetime('2011-10-01'))][['date', 'hashrate', 'price', 'change_30']])
	
	
	btc['dif_change'] = btc['difficulty'].diff().fillna(0)
	btc['hw_change'] = btc['dif_change'] != 0
	
	# Revenue in USD per hour
	btc['revenue'] = btc['price']*((btc['block_count']/24.0)*btc['block_reward'] + btc['tx_fee']/24.0)
	btc['rev_per_hashrate'] = btc['revenue'] / btc['hashrate']
	
	bch['block_count'] = (144 * 600 * 1e12 / 2**32) * (bch['hashrate'] / bch['difficulty'])
	# BCH Revenue in USD per hour
	bch['revenue'] = bch['price']*((bch['block_count']/24.0)*bch['block_reward'])
	bch['rev_per_hashrate'] = bch['revenue'] / bch['hashrate']
	
	b = pd.merge(btc[['date', 'rev_per_hashrate', 'change']], bch[['date', 'rev_per_hashrate', 'change']], on='date')
	btc_to_bch = b[(b['date'] >= pd.to_datetime('2017-08-01')) & (b['rev_per_hashrate_x'] < b['rev_per_hashrate_y']) & (b['change_x'] < 0) & (b['change_y'] > 0)]
	bch_to_btc = b[(b['date'] >= pd.to_datetime('2017-08-01')) & (b['rev_per_hashrate_x'] > b['rev_per_hashrate_y']) & (b['change_x'] > 0) & (b['change_y'] < 0)]	
	
	market_share, new_hw, old_hw, on_again, avg_e_price = model(btc, mining_hw, break_even, btc_to_bch, bch_to_btc)
	
	energy_consumption = pd.read_csv('bitcoin-energy-consumption.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'estimated', 'minimum'])

	energy_consumption['date'] = energy_consumption['date'].astype('datetime64[D]')
	
	y = []
	j = 0
	for i in market_share.loc[:, market_share.columns != 'date'].columns.values:
		y.append(market_share[i]*mining_hw.iloc[j]['power_usage']*24/1000000)
		j += 1
	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.stackplot(btc['date'].values, y, labels=market_share.columns[1:].values)
	ax.plot(energy_consumption['date'].values, (energy_consumption['estimated']*1000/365).values)
	ax.plot(energy_consumption['date'].values, (energy_consumption['minimum']*1000/365).values, color='red')
	ax.set_ylabel('Energy Consumption (GWh)')
	ax.grid()
	ax.legend(loc='upper left')
	plt.show()
	
	bought = pd.DataFrame(new_hw, columns=['date', 'device', 'hashrate'])
	b2 = pd.merge(btc[['date', 'rev_per_hashrate']], bought[['date', 'device', 'hashrate']])
	b3 = pd.merge(b2[['device']], mining_hw[['power_usage']], how='left', left_on='device', right_index=True)
	b2['prof_price'] = b2['rev_per_hashrate']/b3['power_usage']
	b2['weight'] = b2['hashrate']*b3['power_usage']
	buy = b2[~(b2['date'].isin(bch_to_btc['date'].values))]
	
	sold = pd.DataFrame(old_hw, columns=['date', 'device', 'hashrate'])
	s2 = pd.merge(btc[['date', 'rev_per_hashrate']], sold[['date', 'device', 'hashrate']])
	s3 = pd.merge(s2[['device']], mining_hw[['power_usage']], how='left', left_on='device', right_index=True)
	s2['prof_price'] = s2['rev_per_hashrate']/s3['power_usage']
	s2['weight'] = s2['hashrate']*s3['power_usage']
	sell = s2[~(s2['date'].isin(btc_to_bch['date'].values))]
	print(sell['weight'].sum())
	
	reused = pd.DataFrame(on_again, columns=['date', 'device', 'hashrate'])
	r2 = pd.merge(btc[['date', 'rev_per_hashrate']], reused[['date', 'device', 'hashrate']])
	r3 = pd.merge(r2[['device']], mining_hw[['power_usage']], how='left', left_on='device', right_index=True)
	r2['prof_price'] = r2['rev_per_hashrate']/r3['power_usage']
	r2['weight'] = r2['hashrate']*r3['power_usage']
	reuse = r2[~(r2['date'].isin(bch_to_btc['date'].values))]
	
	fig, ax = plt.subplots()
	sc1 = ax.scatter(sell['date'].values, sell['prof_price'].values, c=sell['device'], marker='v', cmap='tab20')
#	sc2 = ax.scatter(buy['date'].values, buy['prof_price'].values, c=buy['device'], marker='^', cmap='tab20')
#	sc3 = ax.scatter(reuse['date'].values, reuse['prof_price'].values, c=reuse['device'], marker='x', cmap='tab20')
#	l1 = ax.legend(*sc2.legend_elements(), loc="upper left", title="Devices")
#	ax.add_artist(l1)
	plt.show()

	y = []
	for i in market_share.loc[:, market_share.columns != 'date'].columns.values:
		y.append(market_share[i])
	
	
	
	fig, ax = plt.subplots()
	ax.set_yscale('log')
#	ax.set_ylim((0, 0.6e8))
	ax.stackplot(btc['date'].values, y, labels=market_share.columns[1:].values, cmap='tab20')
#	ax.plot(btc['date'].values, btc['hashrate'].values, c='b', alpha=0.5)
#	ax2 = ax.twinx()
#	ax2.set_ylim((0, 22000))
#	ax2.set_yscale('linear')
#	ax2.plot(btc['date'].values, btc['price'].values, c='r')
	ax.set_ylabel('Hash Rate (TH/s)')
	ax.grid()
	ax.legend(loc='upper left')
	plt.show()
	
	fig, ax = plt.subplots()
	ax.set_yscale('linear')
	ax.hist(sell['prof_price'].values, bins=100, density=False, weights=sell['weight']/1000, color='r', alpha=0.5)
#	ax.hist(b2['prof_price'].values, bins=100, color='g', alpha=0.5)
	ax.set_ylabel('Power (MW)')
	ax.set_xlabel('Electricity Price (USD)')
	ax.grid()
	plt.show()


if __name__ == "__main__":
	main()
