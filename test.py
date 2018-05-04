# Import af nødvendige packages
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Trækker data fra Quandl
# Min API key til udtræk
quandl.ApiConfig.api_key = "yTPaspmH6wqs9rAdSdmk"
# Valgte aktiver
selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
# Træk af data, WIKI/PRICES er databasen med priser af US stocks fra Quandl
# qopts = "request data from specific columns"
# gte = greater than or equal
# lte = less than or equal
data = quandl.get_table('WIKI/PRICES', ticker = selected, qopts = {'columns': ['date', 'ticker', 'adj_close']},
                        date = {'gte': '2014-1-1', 'lte': '2016-12-31'}, paginate=True)

# sorterer data efter dato
clean = data.set_index('date')
# danner tabel med "ticker" / symbol som kolonner ie. 'CNP', 'F', 'WMT', 'GE' og 'TSLA'
table = clean.pivot(columns='ticker')

# former data om til procentvis ændring i pris fra tid t til t+1
returns_daily = table.pct_change()
# regner årlig ændring, 252 fordi der er omkring (251.89) trading dage på et år
returns_annual = returns_daily.mean() * 252

# danner covariance-matrix ud fra overstående
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 252

port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

num_assets = len(selected)
num_portfolios = 10000

# laver et seed så resultater kan reproduceres
np.random.seed(101)

# simulerer 10000 (num_portfolios) portføljer med tilfældig vægt
for single_portfolio in range(num_portfolios):
    # giver 5 (num_assets) tilfældige vægtninger mellem 0 og 1
    weights = np.random.random(num_assets)
    # normaliserer så sum af vægte = 1
    weights /= np.sum(weights)
    # udregner profit som prikprodukt af vægte og årlig profit
    returns = np.dot(weights, returns_annual)
    # udregner volatilitet
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    # udregner sharpe
    sharpe = returns / volatility

    # indsætter overstående udregninger i vektorerne
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# definerer portfolier som en sammensætning af overstående
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# definerer de enkelte vægte af de valgte aktiver
for i in range(len((selected))):
    symbol = selected[i]
    portfolio[symbol+' Weight'] = [Weight[i] for Weight in stock_weights]

# sætter vores portføljer som dataframe i pandas
df = pd.DataFrame(portfolio)

column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [symbol+' Weight' for symbol in selected]

df = df[column_order]

# finder min vol og max sharp værdier
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# finder portføljerne der stemmer overens med overstående værdier
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
minvar_portfolio = df.loc[df['Volatility'] == min_volatility]

# plotter hele lortet
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=minvar_portfolio['Volatility'], y=minvar_portfolio['Returns'], c='blue', marker='D', s=200)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()