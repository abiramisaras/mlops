import numpy as np
import pandas as pd

# Generate synthetic data: price = 50 * size + noise
np.random.seed(42)
size = np.random.uniform(500, 3000, 100)  # House size in sq ft
noise = np.random.normal(0, 10000, 100)
price = 50 * size + 20000 + noise  # Base price + noise

# Create DataFrame
df = pd.DataFrame({'size': size, 'price': price})

# Save to CSV
df.to_csv('house_prices.csv', index=False)
print("Dataset created: house_prices.csv")