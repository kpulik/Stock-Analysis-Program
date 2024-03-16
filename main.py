
def main():
    
    # Replace with your API key - its free
    API_KEY = "cnps2npr01qgjjvr3fjgcnps2npr01qgjjvr3fk0"
    
    """
    Presents the user with 3 options and runs the corresponding script.
    """
    print("Welcome to the Stock Analysis Tool!")
    print("Select an option:")
    print("1. Calculate Stock Option Price")
    print("2. Visualize Stock Data")
    print("3. Exit Program")

    # Get user input
    choice = input("Enter your choice (1-3): ")


    # Validate user input
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return
    
    # User Input
    stock_tickers = input("Enter multiple stock tickers separated by commas:\n")
    stock_tickers = stock_tickers.split(",") # Split into a list of tickers
    
    def get_stock_data(stock_tickers):
        import requests
        import pandas as pd

        # Get stock data from Finnhub API
        def get_stock_data(symbol):
            # Make a GET request to Finnhub API to retrieve stock data for a given symbol
            response = requests.get(f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}')
            
            # Convert the response to JSON format
            data = response.json()
            return data

        # List of company symbols for which you want to get the stock data
        companies = stock_tickers #['AAPL', 'GOOGL', 'MSFT']

        # Get the stock data for all companies and store it in a DataFrame
        data = pd.DataFrame()
        for company in companies:
            # Retrieve stock data for each company symbol
            company_data = get_stock_data(company)
            
            # Assign the symbol of the company to the data
            company_data['symbol'] = company
            
            # Concatenate the company data to the main DataFrame
            data = pd.concat([data, pd.DataFrame([company_data])], ignore_index=True)

        # Export the data to a CSV file
        data.to_csv('stock_data.csv', index=False)

        '''
        `c`: The closing price of the stock for the trading day.
        `d`: The change in the stock price from the previous day's closing price.
        `dp`: The percentage change in the stock price from the previous day's closing price.
        `h`: The highest price the stock reached during the trading day.
        `l`: The lowest price the stock reached during the trading day.
        `o`: The opening price of the stock for the trading day.
        `pc`: The previous day's closing price.
        `t`: The Unix timestamp representing the trading day.
        `symbol`: The stock symbol or ticker symbol representing the company or security.
        '''
        
    def stock_option_price_calc():
        import numpy as np
        from scipy.stats import norm
        import finnhub

        finnhub_client = finnhub.Client(API_KEY)

        def black_scholes(call_put, S, K, T, r, sigma):
            """
            Calculates the option price using the Black-Scholes-Merton model.

            Args:
                call_put (str): "c" for call option or "p" for put option.
                S (float): Current stock price (obtained from external source).
                K (float): Strike price of the option contract.
                T (float): Time to expiration of the option (in years).
                r (float): Risk-free interest rate.
                sigma (float): Volatility of the underlying stock.

            Returns:
                float: The predicted option price based on the Black-Scholes model.
            """

            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Calculate option price based on call or put
            if call_put == "c":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return price


        # Loop through each ticker
        for stock_ticker in stock_tickers:
            stock_ticker = stock_ticker.strip()  # Remove any extra whitespace

            K = float(input("Enter strike price for " + stock_ticker + ": "))
            T = float(input("Enter time to expiration (in years) for " + stock_ticker + ": "))
            r = float(input("Enter risk-free interest rate for " + stock_ticker + ": "))
            sigma = float(input("Enter volatility for " + stock_ticker + ": "))
            call_put = input("Enter 'c' for call or 'p' for put for " + stock_ticker + ": ").lower()

            # Fetch real-time stock price using Finnhub API
            quote = finnhub_client.quote(stock_ticker)
            current_price = quote['c']

            # Calculate and Output
            predicted_price = black_scholes(call_put, current_price, K, T, r, sigma)
            print("----------------------------------")
            print("The predicted option price for " + stock_ticker + " according to the Black-Scholes-Merton model is:", predicted_price)
            print("The current price of the " + stock_ticker + " stock is:", current_price)

    def stock_data_vistualization():
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the data from CSV file
        data = pd.read_csv('stock_data.csv')

        # Group the data by symbol (stock ticker)
        grouped = data.groupby('symbol')

        # Create a figure and axis objects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Iterate over each group (stock) and plot the data
        for name, group in grouped:
            # Drop the 'symbol' column from the predictors as it may not be relevant
            predictors = group.drop(['o', 'symbol'], axis=1)
            
            # Choose the target variable to predict
            target = group['o']
            
            # Create a linear regression model
            regressor = LinearRegression()
            
            # Train the model using the data for this stock
            regressor.fit(predictors, target)
            
            # Use the trained model to make predictions
            y_pred = regressor.predict(predictors)
            
            # Plot the actual vs. predicted values for this stock
            ax1.scatter(target, y_pred, label=name)

        # Add labels and title for the actual vs. predicted plot
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs. Predicted Values")
        ax1.legend()

        # Plot the feature importances for the last stock
        last_group = data.groupby('symbol').last().reset_index()  # Convert to DataFrame
        predictors = last_group.drop(['o', 'symbol'], axis=1)
        target = last_group['o']
        regressor.fit(predictors, target)
        feature_importances = np.abs(regressor.coef_)
        ax2.bar(range(len(feature_importances)), feature_importances)
        ax2.set_xticks(range(len(feature_importances)))
        ax2.set_xticklabels(predictors.columns, rotation=90)
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Importance")
        ax2.set_title("Feature Importances (Last Stock)")

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.4)

        # Show the plots
        plt.show()

    
    # Run the chosen script based on user input
    if choice == "1":
        stock_option_price_calc()
    elif choice == "2":
        get_stock_data(stock_tickers)
        stock_data_vistualization()
    else:
        print("Goodbye!")

if __name__ == "__main__":
  main()
