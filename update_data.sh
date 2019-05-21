#!/bin/bash
curl https://api.blockchain.info/charts/market-price?timespan=11years\&start=1230800000\&sampled=false\&format=csv > market-price.csv
curl https://api.blockchain.info/charts/hash-rate?timespan=11years\&start=1230800000\&sampled=false\&format=csv > hash-rate.csv
curl https://api.blockchain.info/charts/transaction-fees?timespan=11years\&start=1230800000\&sampled=false\&format=csv > transaction-fees.csv
curl https://api.blockchain.info/charts/n-transactions?timespan=11years\&start=1230800000\&sampled=false\&format=csv > n-transactions.csv
curl https://api.blockchain.info/charts/difficulty?timespan=11years\&start=1230800000\&sampled=false\&format=csv > difficulty.csv
