# %% [markdown]
# # Week 4 Final Project: Grade Tracker ML Prediction
# This notebook loads student data, visualizes it, calculates statistics, 
# and uses Machine Learning to predict a student's future grade.

# %%
# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import os

# %%
# 2. GENERATE MOCK CSV DATA (So the project runs out-of-the-box)
def create_mock_csv(filename="student_grades.csv"):
    if not os.path.exists(filename):
        # Creating past grades for two students over 5 tests
        data = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d")] * 10,
            "Name": ["Alice"]*5 + ["Bob"]*5,
            "ID": [101]*5 + [102]*5,
            "Subject": ["Math"]*5 + ["Science"]*5,
            "Test_Number": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "Grade": [70, 74, 79, 83, 86, 65, 68, 64, 72, 75] 
        }
        temp_df = pd.DataFrame(data)
        temp_df.to_csv(filename, index=False)
        print(f"Created mock data file: {filename}\n")

create_mock_csv()

# %%
# 3. LOAD DATA INTO PANDAS DATAFRAME
print("--- Loading Data ---")
df = pd.read_csv("student_grades.csv")
print(df)

# %%
# 4. SHOW CLASS STATS USING NUMPY
print("--- Class Statistics (Numpy) ---")
grades_array = df['Grade'].to_numpy()

class_average = np.mean(grades_array)
highest_score = np.max(grades_array)
lowest_score = np.min(grades_array)

print(f"Class Average: {class_average:.2f}")
print(f"Highest Score: {highest_score}")
print(f"Lowest Score:  {lowest_score}")

# %%
# 5. GENERATE BAR CHARTS USING MATPLOTLIB
# Group the data to see the average grade per subject
subject_avg = df.groupby('Subject')['Grade'].mean()

plt.figure(figsize=(8, 5))
subject_avg.plot(kind='bar', color=['skyblue', 'lightgreen'], edgecolor='black')
plt.title('Average Grades per Subject', fontsize=14)
plt.xlabel('Subject', fontsize=12)
plt.ylabel('Average Grade', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# 6. TRAIN A LINEAR REGRESSION MODEL & PREDICT NEXT SCORE
print("--- Machine Learning Prediction ---")

# Let's isolate Alice's Math grades to track her progression over time
alice_data = df[(df['Name'] == 'Alice') & (df['Subject'] == 'Math')]

# Define our Features (X: Test Number) and Target (y: Grade)
X = alice_data[['Test_Number']].values  # scikit-learn expects a 2D array for X
y = alice_data['Grade'].values

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Predict Alice's likely score for Test #6
next_test = np.array([[6]])
predicted_score = model.predict(next_test)

print(f"Based on past performance, Alice's predicted score for Test 6 is: {predicted_score[0]:.2f}")

# %%
# 7. VISUALIZE THE ML PREDICTION
plt.figure(figsize=(8, 5))

# Plot actual past grades
plt.scatter(X, y, color='blue', s=100, label='Actual Past Grades')

# Plot the ML trendline
plt.plot(X, model.predict(X), color='red', linestyle='--', label='ML Trendline')

# Plot the future prediction
plt.scatter(next_test, predicted_score, color='green', marker='*', s=200, label='Predicted Test 6')

plt.title("Linear Regression: Predicting Alice's Next Math Grade", fontsize=14)
plt.xlabel("Test Number", fontsize=12)
plt.ylabel("Grade", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()