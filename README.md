# Rare-Disease Case Identification Using DBSCAN-Based Anomaly Detection

## 1. Project Overview

Rare diseases are difficult to identify because their symptoms and patient patterns may differ from common cases.

This project develops a machine learning system that identifies unusual patient patterns using the DBSCAN clustering algorithm.

The system uses patient information such as age, symptoms, symptom duration, family history, and hospital visits.

The project is implemented using Python and Streamlit.

---

## 2. Objectives

The main objectives of this project are:

- To preprocess patient medical data.
- To convert categorical and symptom information into numerical values.
- To apply feature scaling.
- To use DBSCAN for unsupervised anomaly detection.
- To identify unusual patient patterns.
- To allow users to analyze a new patient.
- To allow new patient records to be added to the dataset.
- To compare unusual cases with the existing Rare/Common labels.
- To provide a simple web-based interface using Streamlit.

---

## 3. Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- DBSCAN
- StandardScaler

### Data Processing

- Pandas

### Web Application

- Streamlit

### Development Environment

- Visual Studio Code

---

## 4. Dataset

The dataset contains patient records with the following attributes:

| Feature | Description |
|---|---|
| PatientID | Unique patient identifier |
| Age | Patient age |
| Fatigue | Fatigue severity |
| Fever | Fever severity |
| Weakness | Weakness severity |
| Seizures | Seizure severity |
| Pain | Pain severity |
| Gender | Patient gender |
| Symptom_Duration_Days | Duration of symptoms |
| Family_History | Family history information |
| Hospital_Visits_Last_Year | Number of hospital visits |
| Label | Rare or Common case |

The dataset contains both Rare and Common patient cases.

The Rare/Common label is not used for DBSCAN training.

---

## 5. Data Preprocessing

The symptom columns contain values such as:

7 (Severe)

3 (Mild)

The numeric portion is extracted from these values.

For example:

7 (Severe) → 7

Categorical values are converted into numerical values.

### Gender

Male → 0

Female → 1

### Family History

No → 0

Yes → 1

The numerical features are then standardized using StandardScaler.

---

## 6. Features Used

The following features are provided to the machine learning model:

- Age
- Fatigue
- Fever
- Weakness
- Seizures
- Pain
- Gender
- Symptom Duration
- Family History
- Hospital Visits Last Year

---

## 7. Machine Learning Algorithm

### DBSCAN

DBSCAN stands for:

**Density-Based Spatial Clustering of Applications with Noise**

It is an unsupervised machine learning algorithm.

DBSCAN identifies groups of data points based on their density.

In this project, records that do not belong to dense patient patterns are treated as unusual cases.

The model uses:

```python
DBSCAN(
    eps=1.5,
    min_samples=10
)