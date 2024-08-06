quarterly = pd.read_csv(historical_root + 'historical_quarterly_240320 (1).csv', index_col=0, parse_dates=True) / 100
quarterly = quarterly.dropna(how='all')
monthly = pd.read_csv(historical_root + 'historical_monthly_240319 (1).csv', index_col=0, parse_dates=True) / 100
monthly = monthly.resample('Q').agg(lambda x: np.nan if np.isnan(x).any() else (x + 1).prod() - 1)

merged = pd.merge(monthly, quarterly, left_index=True, right_index=True, how='outer')  # merged is quarterly returns