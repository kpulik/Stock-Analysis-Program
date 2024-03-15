import requests
import pandas as pd

# Get stock data from Finnhub
def get_stock_data(symbol):
    response = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token=cnps2npr01qgjjvr3fjgcnps2npr01qgjjvr3fk0')
    data = response.json()
    return data

# List of company symbols for which you want to get the stock data
companies = ['AAPL', 'GOOGL', 'MSFT']

# Get the stock data for all companies and store it in a DataFrame
data = pd.DataFrame()
for company in companies:
    company_data = get_stock_data(company)
    company_data['symbol'] = company
    data = data.concat(company_data, ignore_index=True)

# Export the data to a CSV file
data.to_csv('stock_data.csv', index=False)
