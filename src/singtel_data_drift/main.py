import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score



train_df = pd.read_csv(
    "/root/naisc/NAISC-Singtel-2026/public_data/train.csv",
    header=0,
)
test_df = pd.read_csv(
    "/root/naisc/NAISC-Singtel-2026/public_data/test.csv",
    header=0,
)


target = 'ChurnStatus'
drop_cols = ['CustomerID', 'Month', target]

X = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')


y = train_df[target].map({'No': 0, 'Yes': 1}) # Map binary target
y_test = test_df[target].map({'No': 0, 'Yes': 1}) # Map binary target

# 2. Split for local validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# 3. Initialize LightGBM with FIXED Hyperparameters
# Using the Scikit-learn API for ease of use
params = {
    'verbosity': -1,
    'objective': 'binary',
    'is_unbalance': True,
    'random_state': 42,
    'importance_type': 'gain'
}

model = lgb.LGBMClassifier(**params)

# 4. Train
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

print("Model Initialized and Trained.")

importances = model.feature_importances_
feature_names = X.columns

# Create a quick dataframe for sorting
feature_stats = pd.DataFrame({'feature': feature_names, 'gain': importances})
feature_stats = feature_stats.sort_values(by='gain', ascending=False)

print(feature_stats.head(10)) # See top 10 most impactful features


# 1. Detailed Report (Precision, Recall, F1)
print("validation: ")
print("Classification report:", classification_report(y_val, model.predict(X_val)))

# 2. ROC-AUC (The most common metric for Churn)
auc_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"AUC: {auc_score:.4f}")

print("Testing: ")
print("Classification report:", classification_report(y_test, model.predict(X_test)))

auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC: {auc_score:.4f}")

print('model.predict_proba(X_test)', model.predict_proba(X_test))



