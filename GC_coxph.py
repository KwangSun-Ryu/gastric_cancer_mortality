import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sksurv.metrics import concordance_index_censored
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. CSV load
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# 2. removing duplicates
X_train = X_train.loc[:, ~X_train.columns.str.lower().duplicated()]
X_test = X_test.loc[:, ~X_test.columns.str.lower().duplicated()]

# 3. Convert event to boolean format (0 -> False, 1 -> True)
y_train['event'] = y_train['event'].apply(lambda x: bool(x))
y_test['event'] = y_test['event'].apply(lambda x: bool(x))

# 4. y Convert data to Surv objects
y_train_structured = Surv.from_dataframe("event", "time", y_train)
y_test_structured = Surv.from_dataframe("event", "time", y_test)

# 5. scaling (using StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Remove traits with very small variance
selector = VarianceThreshold(threshold=1e-5)
X_train_reduced = selector.fit_transform(X_train_scaled)
X_test_reduced = selector.transform(X_test_scaled)

# 7. Extract only the column names kept by the VarianceThreshold
selected_features = X_train.columns[selector.get_support()]

# 8. Remove duplicate attributes (keep column names after converting to DataFrame format)
X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=selected_features).T.drop_duplicates().T
X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=selected_features).T.drop_duplicates().T

# 9. Calculate VIF to remove highly multicollinear features
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


vif_data = calculate_vif(X_train_reduced_df)


high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].values

# Remove the same attribute from X_train and X_test
X_train_final = X_train_reduced_df.drop(columns=[col for col in high_vif_features if col in X_train_reduced_df.columns], errors='ignore')
X_test_final = X_test_reduced_df.drop(columns=[col for col in high_vif_features if col in X_test_reduced_df.columns], errors='ignore')

# 10. run 10 iterations
n_iterations = 10
results = []  
all_c_indices = []

for iteration in range(1, n_iterations + 1):
    # trin the CoxPH model
    coxph = CoxPHSurvivalAnalysis()
    coxph.fit(X_train_final, y_train_structured)

    # C-index 
    predictions = coxph.predict(X_test_final)
    c_index = concordance_index_censored(y_test_structured["event"], y_test_structured["time"], predictions)[0]  # C-index만 사용
    all_c_indices.append(c_index)

    # Feature importance is the coefficient of the CoxPH model (calculates importance in absolute value)
    feature_importances = abs(coxph.coef_)
    
    # Save results for each attribute
    for feature, importance in zip(X_train_final.columns, feature_importances):
        results.append({
            "Iteration": iteration,
            "C-index": c_index,  
            "Feature": feature,  
            "Importance": importance
        })


results_df = pd.DataFrame(results)
results_df.to_csv('coxph_feature_importances.csv', index=False)