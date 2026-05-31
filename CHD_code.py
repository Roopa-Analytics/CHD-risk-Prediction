#%%
import numpy as np
import pandas as pd 

df=pd.read_csv("C:/Users/Admin/Downloads/framingham.csv")
df

df.head()
df.tail()
df.shape
df.isnull().sum()

#drop education columns since it is not require for clinical case 
df=df.drop(columns=["education"])
df.isnull().sum()

#filling missing values by medain for numerical values 
num_cols=['cigsPerDay',"totChol","BMI","heartRate","glucose"]
for col in num_cols:
    df[col]=df[col].fillna(df[col].median())
    
df.isnull().sum()

#filling missing calues for categorical columns
df["BPMeds"]=df["BPMeds"].fillna(df["BPMeds"].mode()[0])
df.isnull().sum()


#%%
#logistic 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# 1. Features and target
feature_cols = ['male','age','currentSmoker','cigsPerDay','BPMeds',
                'prevalentStroke','prevalentHyp','diabetes',
                'totChol','sysBP','diaBP','BMI','heartRate','glucose']

X = df[feature_cols]
y = df['TenYearCHD']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Baseline model: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# 5. Predictions and metrics
y_pred_lr  = lr.predict(X_test_scaled)
y_prob_lr  = lr.predict_proba(X_test_scaled)[:, 1]

acc  = accuracy_score(y_test, y_pred_lr)
prec = precision_score(y_test, y_pred_lr)
rec  = recall_score(y_test, y_pred_lr)
f1   = f1_score(y_test, y_pred_lr)
auc  = roc_auc_score(y_test, y_prob_lr)

print("Logistic Regression performance:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("ROC-AUC:", auc)

# 6. Confusion matrix (numbers)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion matrix (LR):")
print(cm_lr)

TN, FP, FN, TP = cm_lr.ravel()
total = TN + FP + FN + TP
errors = FP + FN
error_rate = errors / total
accuracy_from_cm = (TN + TP) / total

print("Total samples:", total)
print("Errors (FP+FN):", errors)
print("Error rate:", error_rate)
print("Error %:", error_rate * 100)
print("Accuracy (from CM):", accuracy_from_cm)

# 7. Confusion matrix heatmap
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                 display_labels=['No CHD', 'CHD'])
disp_lr.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('cm_lr.png', dpi=300, bbox_inches='tight')
plt.show()



#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. Random Forest model (tune later if needed)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight=None  # or 'balanced' if you want
)
rf.fit(X_train, y_train)  # Note: RF can work on unscaled features

# 2. Predictions and probabilities
y_pred_rf  = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

# 3. Metrics
acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf  = recall_score(y_test, y_pred_rf)
f1_rf   = f1_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, y_prob_rf)

print("Random Forest performance:")
print("Accuracy:", acc_rf)
print("Precision:", prec_rf)
print("Recall:", rec_rf)
print("F1:", f1_rf)
print("ROC-AUC:", auc_rf)

# 4. Confusion matrix (numbers)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion matrix (RF):")
print(cm_rf)

TN_rf, FP_rf, FN_rf, TP_rf = cm_rf.ravel()
total_rf = TN_rf + FP_rf + FN_rf + TP_rf
errors_rf = FP_rf + FN_rf
error_rate_rf = errors_rf / total_rf
accuracy_from_cm_rf = (TN_rf + TP_rf) / total_rf

print("Total samples (RF):", total_rf)
print("Errors (FP+FN) (RF):", errors_rf)
print("Error rate (RF):", error_rate_rf)
print("Error % (RF):", error_rate_rf * 100)
print("Accuracy (from CM) (RF):", accuracy_from_cm_rf)

# 5. Confusion matrix heatmap
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf,
                                 display_labels=['No CHD', 'CHD'])
disp_rf.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('cm_rf.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. XGBoost model (basic setup)
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

# You can use scaled or unscaled features; start with unscaled for trees:
xgb_model.fit(X_train, y_train)

# 2. Predictions and probabilities
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 3. Metrics
acc_xgb  = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb  = recall_score(y_test, y_pred_xgb)
f1_xgb   = f1_score(y_test, y_pred_xgb)
auc_xgb  = roc_auc_score(y_test, y_prob_xgb)

print("XGBoost performance:")
print("Accuracy:", acc_xgb)
print("Precision:", prec_xgb)
print("Recall:", rec_xgb)
print("F1:", f1_xgb)
print("ROC-AUC:", auc_xgb)

# 4. Confusion matrix (numbers)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("Confusion matrix (XGB):")
print(cm_xgb)

TN_xgb, FP_xgb, FN_xgb, TP_xgb = cm_xgb.ravel()
total_xgb = TN_xgb + FP_xgb + FN_xgb + TP_xgb
errors_xgb = FP_xgb + FN_xgb
error_rate_xgb = errors_xgb / total_xgb
accuracy_from_cm_xgb = (TN_xgb + TP_xgb) / total_xgb

print("Total samples (XGB):", total_xgb)
print("Errors (FP+FN) (XGB):", errors_xgb)
print("Error rate (XGB):", error_rate_xgb)
print("Error % (XGB):", error_rate_xgb * 100)
print("Accuracy (from CM) (XGB):", accuracy_from_cm_xgb)

# 5. Confusion matrix heatmap
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb,
                                  display_labels=['No CHD', 'CHD'])
disp_xgb.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - XGBoost')
plt.savefig('cm_xgb.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
#visuals
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Probabilities already: y_prob (LR), y_prob_rf, y_prob_xgb

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC = {auc:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC = {auc_rf:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for CHD Risk Prediction Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_models.png', dpi=300)
plt.show()

#.plot 2
import numpy as np
import matplotlib.pyplot as plt

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(7,5))
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), np.array(feature_cols)[indices], rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300)
plt.show()



#.plot 3
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
sns.boxplot(x='TenYearCHD', y='age', data=df)
plt.xlabel('TenYearCHD (0=no, 1=yes)')
plt.ylabel('Age')
plt.title('Age distribution by CHD outcome')
plt.tight_layout()
plt.savefig('age_CHD_boxplot.png', dpi=300)
plt.show()


#confusion matrix for Logistic 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_lr = confusion_matrix(y_test, y_pred_lr)

disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                 display_labels=['No CHD', 'CHD'])
disp_lr.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('cm_lr.png', dpi=300, bbox_inches='tight')
plt.show()



#%%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ---------- 1. CLASS DISTRIBUTION PLOT ----------
plt.figure(figsize=(4,3))
y_test.value_counts().rename({0:'No CHD', 1:'CHD'}).plot(kind='bar')
plt.ylabel("Count")
plt.title("10-year CHD outcome")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 2. ROC CURVES ----------
y_proba_lr  = log_reg.predict_proba(X_test_scaled)[:,1]  # if you used scaling
y_proba_rf  = rf_clf.predict_proba(X_test)[:,1]
y_proba_xgb = xgb_clf.predict_proba(X_test)[:,1]

fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_proba_lr)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

auc_lr  = roc_auc_score(y_test, y_proba_lr)
auc_rf  = roc_auc_score(y_test, y_proba_rf)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

plt.figure(figsize=(4,4))
plt.plot(fpr_lr,  tpr_lr,  label=f"Logistic (AUC={auc_lr:.3f})")
plt.plot(fpr_rf,  tpr_rf,  label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
plt.plot([0,1],[0,1],'k--',label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 3. METRICS BAR CHART + CONFUSION MATRICES ----------
import numpy as np

# plug in YOUR numbers (you already computed these)
models = ["Logistic", "RandomF", "XGBoost"]
acc  = [0.8453, 0.8472, 0.8443]
prec = [0.4286, 0.4762, 0.4524]
rec  = [0.0559, 0.0621, 0.1180]
f1   = [0.0989, 0.1099, 0.1872]
auc  = [0.7044, 0.6656, 0.6724]

x = np.arange(len(models))
width = 0.16

plt.figure(figsize=(6,4))
plt.bar(x - 2*width, acc,  width, label="Accuracy")
plt.bar(x - width,    prec, width, label="Precision")
plt.bar(x,            rec,  width, label="Recall")
plt.bar(x + width,    f1,   width, label="F1-score")
plt.bar(x + 2*width,  auc,  width, label="AUC")
plt.xticks(x, models)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Model performance metrics")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("metrics_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# Combined confusion matrices using your counts
cms = {
    "Logistic Regression": np.array([[887, 12],
                                     [152,  9]]),
    "Random Forest":       np.array([[888, 11],
                                     [151, 10]]),
    "XGBoost":             np.array([[876, 23],
                                     [142, 19]])
}

fig, axes = plt.subplots(1, 3, figsize=(9,3))

for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["No CHD", "CHD"], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(["No CHD", "CHD"], rotation=0, fontsize=7)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()
