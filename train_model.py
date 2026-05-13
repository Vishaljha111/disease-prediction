import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ================================
# STEP 1: Load Dataset
# ================================
print("📂 Loading dataset...")
df = pd.read_csv('dataset/diabetes.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic Stats:")
print(df.describe())

# ================================
# STEP 2: Data Visualization
# ================================
print("\n📊 Creating visualizations...")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.close()
print("✅ Heatmap saved!")

plt.figure(figsize=(6, 4))
df['Outcome'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Diabetes Outcome Distribution')
plt.xlabel('Outcome (0=No, 1=Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outcome_distribution.png')
plt.close()
print("✅ Outcome chart saved!")

# ================================
# STEP 3: Prepare Data
# ================================
print("\n⚙️ Preparing data...")

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ================================
# STEP 4: Train Multiple Models
# ================================
print("\n🤖 Training models...")

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42, probability=True)
}

results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = acc
    print(f"✅ {name}: {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy:.2f}% accuracy")

# ================================
# STEP 5: Model Comparison Chart
# ================================
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(60, 100)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("✅ Model comparison chart saved!")

# ================================
# STEP 6: Confusion Matrix
# ================================
y_pred_best = best_model.predict(X_test_scaled)
print(f"\n📋 Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best))

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved!")

# ================================
# STEP 7: Save Model & Scaler
# ================================
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("\n✅ Model and scaler saved!")
print("🎉 Training complete!")