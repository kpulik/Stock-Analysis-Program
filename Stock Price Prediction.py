import numpy as np
import finnhub

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
        price = S * np.norm.cdf(d1) - K * np.exp(-r * T) * np.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * np.norm.cdf(-d2) - S * np.norm.cdf(-d1)

    return price

# User Input (replace with actual prompts and data collection)
stock_ticker = "AAPL"  # Replace with the desired stock ticker
K = float(input("Enter strike price: "))  # Strike price
T = float(input("Enter time to expiration (in years): "))  # Time to expiration (in years)
r = float(input("Enter risk-free interest rate: "))  # Risk-free interest rate
sigma = float(input("Enter volatility: "))  # Volatility
call_put = input("Enter 'c' for call or 'p' for put: ").lower()  # "c" for call or "p" for put

# Fetch real-time stock price using Finnhub API
finnhub_client = finnhub.Client(api_key="cnps2npr01qgjjvr3fjgcnps2npr01qgjjvr3fk0")  # Replace with your API key
quote = finnhub_client.quote(stock_ticker)
current_price = quote['c']

# Calculate and Output
predicted_price = black_scholes(call_put, current_price, K, T, r, sigma)
print("The predicted option price according to the Black-Scholes-Merton model is:", predicted_price)
