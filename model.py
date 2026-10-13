import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load data
data = pd.read_csv("data.csv", sep="\t")

# Convert symptom values to numbers
symptoms = ["Fatigue", "Fever", "Weakness", "Seizures", "Pain"]

for column in symptoms:
    data[column] = data[column].str.extract(r"(\d+)").astype(float)

# Convert Gender
data["Gender"] = data["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Convert Family History
data["Family_History"] = data["Family_History"].map({
    "No": 0,
    "Yes": 1
})

# Select data for DBSCAN
features = [
    "Age",
    "Fatigue",
    "Fever",
    "Weakness",
    "Seizures",
    "Pain",
    "Gender",
    "Symptom_Duration_Days",
    "Family_History",
    "Hospital_Visits_Last_Year"
]

X = data[features]

# Scale data
X = StandardScaler().fit_transform(X)

# DBSCAN
model = DBSCAN(eps=1.5, min_samples=10)

data["Cluster"] = model.fit_predict(X)

# Show result
print("DBSCAN completed!")

print("\nCluster results:")
print(data["Cluster"].value_counts())

# Show anomalies
anomalies = data[data["Cluster"] == -1]

print("\nPossible anomalies:", len(anomalies))

print("\nAnomaly patients:")
print(anomalies[["PatientID", "Age", "Label"]].head(10))