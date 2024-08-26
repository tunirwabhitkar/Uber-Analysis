import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# URL of the dataset
url = "https://drive.google.com/uc?export=download&id=1XuZ6dwQqeccatMuu6veh5miFjhdWh9I0"

# Load dataset
dataset = pd.read_csv(url)

# Data Preprocessing
# Fill missing values in 'PURPOSE' with 'NOT'
dataset['PURPOSE'] = dataset['PURPOSE'].fillna("NOT")

# Convert START_DATE and END_DATE to datetime format
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')

# Extract date and time from START_DATE
dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

# Convert time to categories: Morning, Afternoon, Evening, Night
dataset['day-night'] = pd.cut(x=dataset['time'],
                              bins=[0,10,15,19,24],
                              labels=['Morning','Afternoon','Evening','Night'])

# Drop rows with null values
dataset.dropna(inplace=True)

# Drop duplicate rows
dataset.drop_duplicates(inplace=True)

# Data Visualization
# Unique values in object datatype columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
unique_values = {col: dataset[col].unique().size for col in object_cols}
print("Unique values in object columns:", unique_values)

# Plot count of CATEGORY and PURPOSE
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)
plt.show()

# Plot count of time categories
sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)
plt.show()

# Compare CATEGORY and PURPOSE
plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

# One-hot encode 'CATEGORY' and 'PURPOSE'
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)

# Filter only numeric columns for correlation heatmap
numeric_dataset = dataset.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Visualization of monthly data
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)

mon = dataset.MONTH.value_counts(sort=False)
df = pd.DataFrame({"MONTHS": mon.index, "VALUE COUNT": dataset.groupby('MONTH', sort=False)['MILES'].max()})

plt.figure(figsize=(12, 6))
p = sns.lineplot(data=df, x="MONTHS", y="VALUE COUNT")
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.xticks(rotation=45)
plt.show()

# Visualization of days data
dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
dataset['DAY'] = dataset['DAY'].map(day_label)
day_label_counts = dataset.DAY.value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=day_label_counts.index, y=day_label_counts.values)
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

# Explore the MILES column
plt.figure(figsize=(12, 6))
sns.boxplot(x=dataset['MILES'])
plt.xlabel('MILES')
plt.show()

# Zoom in for MILES < 100
plt.figure(figsize=(12, 6))
sns.boxplot(x=dataset[dataset['MILES'] < 100]['MILES'])
plt.xlabel('MILES')
plt.show()

# Distribution plot for MILES < 40
plt.figure(figsize=(12, 6))
sns.histplot(dataset[dataset['MILES'] < 40]['MILES'], kde=True)
plt.xlabel('MILES')
plt.show()
