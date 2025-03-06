import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('error_params.csv')

# Compute the sum of the last two columns
data['Error_Sum'] = data['Error_World'] + data['Error_Local']

# Define predictor variables (exclude the last two and the sum columns)
predictors = data.columns.drop(['Error_World', 'Error_Local', 'Error_Sum'])
X = data[predictors]
y = data['Error_Sum']

# Fit the regression model
model = LinearRegression()
model.fit(X, y)

# Get and normalize the coefficients
coefficients = pd.Series(model.coef_, index=predictors)
coefficients_normalized = coefficients / coefficients.abs().max()

# Plot the normalized coefficients
coefficients_normalized.plot(kind='bar')
plt.ylabel('Normalized Coefficient')
plt.title('Parameter Significance')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.show()