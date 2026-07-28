
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import os


def create_mock_csv(filename="student_grades.csv"):
    if not os.path.exists(filename):
        
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


print("--- Loading Data ---")
df = pd.read_csv("student_grades.csv")
print(df)


print("--- Class Statistics (Numpy) ---")
grades_array = df['Grade'].to_numpy()

class_average = np.mean(grades_array)
highest_score = np.max(grades_array)
lowest_score = np.min(grades_array)

print(f"Class Average: {class_average:.2f}")
print(f"Highest Score: {highest_score}")
print(f"Lowest Score:  {lowest_score}")


subject_avg = df.groupby('Subject')['Grade'].mean()

plt.figure(figsize=(8, 5))
subject_avg.plot(kind='bar', color=['skyblue', 'lightgreen'], edgecolor='black')
plt.title('Average Grades per Subject', fontsize=14)
plt.xlabel('Subject', fontsize=12)
plt.ylabel('Average Grade', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


print("--- Machine Learning Prediction ---")


alice_data = df[(df['Name'] == 'Alice') & (df['Subject'] == 'Math')]


X = alice_data[['Test_Number']].values  
y = alice_data['Grade'].values


model = LinearRegression()
model.fit(X, y)


next_test = np.array([[6]])
predicted_score = model.predict(next_test)

print(f"Based on past performance, Alice's predicted score for Test 6 is: {predicted_score[0]:.2f}")


plt.figure(figsize=(8, 5))

#  past grades
plt.scatter(X, y, color='blue', s=100, label='Actual Past Grades')


plt.plot(X, model.predict(X), color='red', linestyle='--', label='ML Trendline')

# future prediction
plt.scatter(next_test, predicted_score, color='green', marker='*', s=200, label='Predicted Test 6')

plt.title("Linear Regression: Predicting Alice's Next Math Grade", fontsize=14)
plt.xlabel("Test Number", fontsize=12)
plt.ylabel("Grade", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()