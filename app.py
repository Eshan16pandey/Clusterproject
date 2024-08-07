# Save the provided app.py code to a file

app_code = """
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the models and scaler
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
kmeans = joblib.load('kmeans.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Preprocessing
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['Age'] = 2024 - df['Year_Birth']
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df['Total_Spending'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                            df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['Total_Purchases'] = (df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] +
                             df['NumStorePurchases'])
    
    features = ['Income', 'Age', 'Total_Children', 'Total_Spending', 'Total_Purchases', 'Recency']
    X = df[features]
    
    # Scale and PCA transform
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Predict cluster
    cluster = kmeans.predict(X_pca)
    
    return jsonify({'cluster': cluster.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
"""

# Save the code to a file
file_path = "/mnt/data/app.py"
with open(file_path, "w") as f:
    f.write(app_code)

file_path
