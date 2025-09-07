import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📈 SPY Tomorrow Predictor")
st.write("Logistic Regression models on SPY data (10 years).")

# Step 1: Data Collection
spy = yf.Ticker("SPY")
data = spy.history(period="10y")

data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
dataset = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 2: Feature Engineering
dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1).mean()
dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1).mean()

# Target variables
dataset['Sign_1'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
dataset['Sign_1'] = dataset['Sign_1'].shift(-10)
dataset['Sign_2'] = (np.sign(np.log(dataset['Close'] / dataset['Close'].shift(1))) > 0).astype(int)
dataset['Sign_2'] = dataset['Sign_2'].shift(-1)

# Additional features
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['Log_Return'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
dataset['Momentum'] = dataset['Close'].diff(1)
dataset['SMA_5'] = dataset['Close'].rolling(window=5).mean()
dataset['SMA_10'] = dataset['Close'].rolling(window=10).mean()
dataset['SMA_20'] = dataset['Close'].rolling(window=20).mean()
dataset['SMA_50'] = dataset['Close'].rolling(window=50).mean()
dataset['SMA_100'] = dataset['Close'].rolling(window=100).mean()
dataset['EMA_5'] = dataset['Close'].ewm(span=5, adjust=False).mean()
dataset['EMA_10'] = dataset['Close'].ewm(span=10, adjust=False).mean()
dataset['EMA_20'] = dataset['Close'].ewm(span=20, adjust=False).mean()
dataset['EMA_50'] = dataset['Close'].ewm(span=50, adjust=False).mean()
dataset['EMA_100'] = dataset['Close'].ewm(span=100, adjust=False).mean()

# RSI
delta = dataset['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
dataset['RSI'] = 100 - (100 / (1 + rs))

dataset.dropna(inplace=True)

# Step 3: Train / Validation Split
validation_size = 0.2
split_index = int(len(dataset) * (1 - validation_size))
train_df = dataset.iloc[:split_index]
validation_df = dataset.iloc[split_index:]

features = [
    'O-C', 'H-L', 'Log_Return', 'Momentum',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
    'RSI'
]

X_train = train_df[features]
Y_train_1 = train_df['Sign_1']
Y_train_2 = train_df['Sign_2']

X_validation = validation_df[features]
Y_validation_1 = validation_df['Sign_1']
Y_validation_2 = validation_df['Sign_2']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

# Step 4: Train Models
model1 = LogisticRegression(solver='liblinear')
model1.fit(X_train_scaled, Y_train_1)

model2 = LogisticRegression(solver='liblinear')
model2.fit(X_train_scaled, Y_train_2)

# Step 5: Validation Scores
predictions1 = model1.predict(X_validation_scaled)
predictions2 = model2.predict(X_validation_scaled)

acc1 = accuracy_score(Y_validation_1, predictions1)
acc2 = accuracy_score(Y_validation_2, predictions2)

st.subheader("📊 Model Accuracy")
st.write(f"Model 1 (MA crossover): **{acc1:.2%}**")
st.write(f"Model 2 (Daily returns): **{acc2:.2%}**")

# Step 6: Predict Tomorrow
latest_features = dataset[features].iloc[-1:].copy()
latest_scaled = scaler.transform(latest_features)

# Predictions & probabilities
tomorrow_pred1 = model1.predict(latest_scaled)[0]
tomorrow_prob1 = model1.predict_proba(latest_scaled)[0][1]

tomorrow_pred2 = model2.predict(latest_scaled)[0]
tomorrow_prob2 = model2.predict_proba(latest_scaled)[0][1]

signal_map = {0: "DOWN", 1: "UP"}

st.subheader("🔮 Tomorrow's Prediction")
st.write(f"Model 1 (MA Crossover): **{signal_map[tomorrow_pred1]}** (prob={tomorrow_prob1:.2f})")
st.write(f"Model 2 (Daily Returns): **{signal_map[tomorrow_pred2]}** (prob={tomorrow_prob2:.2f})")

# Step 7: Consensus Signal
votes = [tomorrow_pred1, tomorrow_pred2]
vote_sum = sum(votes)

if vote_sum == 2:
    final_signal = "UP ✅"
elif vote_sum == 0:
    final_signal = "DOWN ⬇️"
else:
    final_signal = "NEUTRAL ⚖️"

st.subheader("📌 Final Consensus Signal")
st.write(f"**{final_signal}**")

# Optional: ROC curve for Model 1
y_pred_prob = model1.predict_proba(X_validation_scaled)[:, 1]
fpr, tpr, _ = roc_curve(Y_validation_1, y_pred_prob)
roc_auc = roc_auc_score(Y_validation_1, y_pred_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (Model 1)')
ax.legend(loc="lower right")
st.pyplot(fig)
