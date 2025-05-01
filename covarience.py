import pandas as pd
import matplotlib.pyplot as plt

# === Adjust this path if your CSV lives elsewhere ===
file_path = 'covariance_data.csv'

# Load the data
data = pd.read_csv(file_path)

# Parameters to compare with Translation_Error
param_cols = [
    'Number of Jumps',
    'Tag_Est_X',
    'Tag_Est_Y',
    'Tag_Est_Z',
    'Tag_Est_Roll',
    'Tag_Est_Pitch',
    'Tag_Est_Yaw'
]

# Compute covariance of each parameter with Translation_Error
cov_values = [
    data[[col, 'Translation_Error']].cov().iloc[0, 1] for col in param_cols
]

# Create a DataFrame for easy viewing
cov_df = pd.DataFrame({
    'Parameter': param_cols,
    'Covariance with Translation_Error': cov_values
})

# Print the covariance values
print(cov_df.to_string(index=False))

# Plot the covariance values
plt.figure()
plt.bar(cov_df['Parameter'], cov_df['Covariance with Translation_Error'])
plt.ylabel('Covariance with Translation Error')
plt.title('Covariance between Parameters and Translation Error')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
