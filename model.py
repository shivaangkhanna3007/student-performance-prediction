import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the CSV file
df = pd.read_csv("data.csv")

# 2. Convert Pass/Fail to numbers
df["Final_Result"] = df["Final_Result"].map({"Pass": 1, "Fail": 0})

# 3. Separate features (X) and target (y)
X = df.drop("Final_Result", axis=1)
y = df["Final_Result"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Test the model
y_pred = model.predict(X_test)

# 7. Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 8. Detailed performance report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Predict for a new student
new_student = [[85, 3.5, 78, 8, 80]]
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nNew Student Prediction: PASS")
else:
    print("\nNew Student Prediction: FAIL")
