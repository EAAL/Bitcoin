from dateutil import parser as parser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Mining hardware data: Extracted manually from multiple sources
# Columns: device (string), release_date (date), hashrate (TH/s), power_usage (W/GH), price (USD)
mining_hw = pd.read_csv('mining_hw.csv', sep=",", skipinitialspace=True, parse_dates=True, comment="#")
mining_hw['release_date'] = mining_hw['release_date'].astype('datetime64[D]')

# Bitcoin prices, hash rates, and transaction fees: From Blockchain.info on 2019-03-19
btc_price = pd.read_csv('market-price.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'price'])
btc_hashrate = pd.read_csv('hash-rate.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'hashrate'])
btc_tx_fees = pd.read_csv('transaction-fees.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'fee'])
btc_txs = pd.read_csv('n-transactions.csv', sep=",", skipinitialspace=True, parse_dates=True, names=['date', 'n'])

btc_price['date'] = btc_price['date'].astype('datetime64[D]')
btc_hashrate['date'] = btc_hashrate['date'].astype('datetime64[D]')
btc_tx_fees['date'] = btc_tx_fees['date'].astype('datetime64[D]')
btc_txs['date'] = btc_txs['date'].astype('datetime64[D]')

# Smoothen the hashrate
w = 10 # window size
btc_hashrate['smooth'] = btc_hashrate['hashrate'].rolling(w, min_periods=w//2, center=True).mean()
# Calculate changes in hash rate
btc_hashrate['change'] = btc_hashrate['smooth'].diff()
btc_hashrate['change'] = btc_hashrate['change'].fillna(0.0)


# Market Share: estimated TH/s for each HW, assuming all mining HW are Antminers (after S1)
cols = ['date'] + mining_hw['device'].values.tolist()
initial_row = [pd.to_datetime('2009-01-01 00:00:00')] + [0 for i in range(len(cols)-1)]
market_share = pd.DataFrame(np.array([initial_row]), columns=cols)

# Modelling the HW market

## Model 1: Assuming the new hardware is bought in increasing hash rates and the oldest gets removed in the decreases
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

# Block rewards in USD per block (halvings are hard-coded)
conditions = [(btc_price['date'] < pd.to_datetime('2012-11-28 00:00:00')),
(btc_price['date'] < pd.to_datetime('2016-07-09 00:00:00')) & (btc_price['date'] >= pd.to_datetime('2012-11-28 00:00:00')),
(btc_price['date'] > pd.to_datetime('2016-07-09 00:00:00'))]
choices = [50.0, 25.0, 12.5]
btc_price['block_reward'] = np.select(conditions, choices, default=0.0)

# Revenue in USD per day
btc_rev = pd.DataFrame([], columns=['date', 'revenue'])
btc_rev['date'] = btc_price['date']
btc_rev['revenue'] = btc_price['price']*(24*6*btc_price['block_reward'] + btc_tx_fees['fee']*btc_txs['n'])

# Cost in USD per day
electricity_price = 0.08 # Price in USD/KWh
btc_cost = pd.DataFrame([], columns=['date', 'cost'])
btc_cost['date'] = market_share['date']
btc_cost['cost'] = (market_share.loc[:, market_share.columns != 'date'].dot(mining_hw['power_usage'].values))*3600*24*electricity_price

cost_share = (market_share.loc[:, market_share.columns != 'date']*mining_hw['power_usage'].values)*3600*24*electricity_price

fig, ax = plt.subplots()
ax.set_yscale('log')
y = []
for i in cost_share.columns.values:
	y.append(cost_share[i])
ax.stackplot(btc_cost['date'].values, y, labels=cost_share.columns.values)
ax.plot(btc_cost['date'], btc_cost['cost'], color='r')
ax.plot(btc_rev['date'], btc_rev['revenue'], color='g')
ax.legend(loc='upper left')
plt.show()
