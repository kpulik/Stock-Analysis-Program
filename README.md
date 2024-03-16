
# Stock Analysis Tool

This program is designed to provide users with stock analysis functionalities, including calculating stock option prices, visualizing stock data, and more. The tool interacts with the Finnhub API to retrieve real-time stock information and performs various analyses based on user input.

## Usage

To use the Stock Analysis Tool, follow the steps below:

1. **API Key Setup:**

   - Before running the program, make sure to replace the placeholder `API_KEY` with your own API key from Finnhub. You can sign up for a free API key on the Finnhub website.
2. **Options Menu:**

   - Upon running the program, you will be presented with the following options:
     - Calculate Stock Option Price
     - Visualize Stock Data
     - Exit Program
3. **User Input:**

   - Enter your choice by typing the corresponding number (1, 2, or 3).
   - For calculating stock option prices, enter multiple stock tickers separated by commas.
   - For visualizing stock data, stock data will be retrieved based on the provided tickers.
4. **Functionality:**

   - **Calculate Stock Option Price:** Calculates the option price using the Black-Scholes-Merton model for the specified stock tickers.
   - **Visualize Stock Data:** Retrieves stock data from the Finnhub API, performs linear regression analysis, and visualizes the data.

## Code Structure

The program consists of the following key functions:

- **`get_stock_data(stock_tickers):`**

  - Retrieves stock data for the specified stock tickers using the Finnhub API and saves the data to a CSV file.
- **`stock_option_price_calc():`**

  - Calculates option prices using the Black-Scholes-Merton model based on user input for strike price, time to expiration, risk-free interest rate, and volatility.
- **`stock_data_visualization():`**

  - Visualizes stock data by performing linear regression and plotting actual vs. predicted values, as well as feature importances.

## Running the Program

To run the Stock Analysis Tool:

1. Make sure you have Python installed.
2. Install the required libraries by running `pip install pandas requests scipy matplotlib numpy scikit-learn finnhub-python` (Ideally in a virtual environment).
3. Replace `API_KEY` with your Finnhub API key.
4. Run the script and follow the on-screen instructions to utilize the different functionalities.

Feel free to explore and analyze stock data efficiently with this tool!
