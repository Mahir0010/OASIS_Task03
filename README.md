# OASIS_Task03
SALE PREDICTION USING PYTHON
ImportLibraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Data collection
data=pd.read_csv("/kaggle/input/advertisingcsv/Advertising.csv",index_col='Unnamed: 0')

EDA
data.head()
data.isnull().sum()
data.describe()
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.7)
plt.show()
sns.heatmap(data.corr(), annot=True)
plt.show()

Training
sns.heatmap(data.corr(), annot=True)
plt.show()
# Define features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

Model Evalute
# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

Testing
# You can now use the trained model to predict sales for new data
new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [10]})
predicted_sales = model.predict(new_data)
print('Predicted Sales:', predicted_sales)

END






