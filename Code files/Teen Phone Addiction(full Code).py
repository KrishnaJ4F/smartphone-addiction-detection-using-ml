import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,precision_score,recall_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("phone_addiction.csv")
df.head()

df.tail()

df.info()

df.describe(include="all")

df = df.drop(columns=['ID', 'Name', 'Phone_Usage_Purpose','Social_Interactions','Parental_Control'])

df["Addiction_Level_Class"] = pd.cut(df["Addiction_Level"],bins=[0, 4, 7, 10],labels=["Low", "Medium", "High"],duplicates="drop")

print("Number of duplicate rows:", df.duplicated().sum())

print("\nUnique Values per Column:\n", df.nunique())

# Check for missing values
print(f" Dataset Shape: {df.shape}")
df.isnull().sum()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
               np.where(df[col] > upper, upper, df[col]))
    
df[num_cols].hist(figsize=(30,20), bins=20, color='skyblue', edgecolor='black')
plt.suptitle('Numerical Feature Distributions')
plt.show()

# Bar plot for a categorical feature (example: Gender)
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Daily_Usage_Hours"], kde=True, bins=30)
plt.title("Distribution of Daily Phone Usage Hours")
plt.xlabel("Daily Phone Usage (hours)")
plt.ylabel("Count")
plt.show()

df["Anxiety_Level"].value_counts().sort_index().plot(kind="bar", figsize=(7,4))
plt.title("Distribution of Anxiety Levels")
plt.xlabel("Anxiety Level")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Sleep_Hours"])
plt.title("Boxplot of Sleep Hours")
plt.xlabel("Sleep Hours")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Daily_Usage_Hours"],y=df["Academic_Performance"])
plt.title("Daily Usage vs Academic Performance")
plt.xlabel("Daily Usage Hours")
plt.ylabel("Academic Performance")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x="Gender", y="Addiction_Level", data=df)
plt.title("Average Addiction Level by Gender")
plt.xlabel("Gender")
plt.ylabel("Addiction Level")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x="Anxiety_Level", y="Sleep_Hours", data=df)
plt.title("Sleep Hours by Anxiety Level")
plt.xlabel("Anxiety Level")
plt.ylabel("Sleep Hours")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Daily_Usage_Hours", y="Academic_Performance",hue="Sleep_Hours",size="Addiction_Level",data=df, palette="viridis")
plt.title("Usage, Sleep & Addiction vs Academic Performance")
plt.xlabel("Daily Usage Hours")
plt.ylabel("Academic Performance")
plt.show()

features = ["Daily_Usage_Hours", "Sleep_Hours", "Anxiety_Level", "Academic_Performance"]

plt.figure(figsize=(7,4))
sns.heatmap(df[features].corr(), annot=True, cmap="Blues")
plt.title("Correlation Between Usage, Sleep, Anxiety & Academics")
plt.show()

le = LabelEncoder()
y = le.fit_transform(df["Addiction_Level_Class"])
X = df.drop(columns=["Addiction_Level_Class"])
cat_cols = X.select_dtypes(include=['object','category']).columns
num_cols = X.select_dtypes(include=['int64','float64']).columns

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30,stratify=y)

sm = SMOTE(random_state=30)
X_train, y_train = sm.fit_resample(X_train, y_train)

log_reg = LogisticRegression(max_iter=50, C=0.01,class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

svm = SVC(kernel="linear", C=0.2)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

knn = KNeighborsClassifier(n_neighbors=5, weights = 'uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

decision = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=30)
decision.fit(X_train, y_train)
y_pred = decision.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=30)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Try XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False
    print("XGBoost not installed. Skipping it.")
    
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric="mlogloss",
    learning_rate=0.2,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=4,
    reg_alpha=4,
    n_estimators=50,
    random_state=30
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(eval_metric="mlogloss")

results = {}

for name, model in models.items():
    print("\n===============================")
    print(f"Training {name}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Store results
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

    # Print metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f" F1 Score : {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    best_model = max(results, key=lambda x: results[x]["accuracy"])
print("BEST MODEL BASED ON ACCURACY",best_model)
print("Accuracy:", results[best_model]["accuracy"])

# Convert results dictionary → DataFrame
results_df = pd.DataFrame([
    {
        "Model": model,
        "Accuracy": results[model]["accuracy"],
        "Precision": results[model]["precision"],
        "Recall": results[model]["recall"],
        "F1 Score": results[model]["f1"]
    }
    for model in results
])

# Sort models by accuracy (best first)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print(results_df)

plt.figure(figsize=(8,5))
plt.bar(results_df['Model'], results_df['Accuracy'], color='skyblue', edgecolor='black')
plt.title("Model Accuracy Comparison", fontsize=14)
plt.xlabel("Machine Learning Models", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=30, ha='right')

# Highlight best model
best_model = results_df.iloc[0]
plt.bar(best_model['Model'], best_model['Accuracy'], color='limegreen', edgecolor='black')

# Annotate accuracy values on top of bars
for i, row in results_df.iterrows():
    plt.text(i, row['Accuracy'] + 0.005, f"{row['Accuracy']:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print(f"\n Best Model: {best_model['Model']} with Accuracy = {best_model['Accuracy']:.4f}")