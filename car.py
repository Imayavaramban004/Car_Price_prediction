import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error, r2_score
db=pd.read_csv("C:\\Users\\imaya\\Downloads\\diamonds.csv")
print(db.head)
target=db["price"]
features=db[["carat","depth"]]
plt.scatter(db["carat"],db["price"])
# plt.show()
x1_train, x1_test, y1_train, y1_test = TTS(features, target, test_size=0.3, random_state=42)
# Train a Linear Regression model
model = LinearRegression()
model1_fit = model.fit(x1_train, y1_train)

# Predict
y1_predict = model1_fit.predict(x1_test)

# Evaluate performance (using regression metrics)
print("Mean Squared Error: ", mean_squared_error(y1_test, y1_predict))
print("R^2 Score: ", r2_score(y1_test, y1_predict))