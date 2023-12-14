# Cardiac_diseases_diagnosis_NLP

## Brief description

The project consists of three python files:
- **main.py** - contains the main logic of the program, code that handles command line arguments
- **preprocessing.py** - contains the MedicalData class, whose methods are used to implement data cleaning and data preprocessing
- **classification.py** - contains the OptunaClassification class, whose methods are used to train both classical machine learning models and neural networks using the Optuna hyperparameter optimization tool.

The dataset used for this project contains four columns of unstructured text, which include the entire record of hospitalization and a column with the diagnosis. These columns have the following names:
- 'History - Onset of disease - Content,
- 'History - Physical examination - Content',
- 'Epicrisis - Physical examination - Content',
- 'Epicrisis - Medical recommendations - Content',
- 'Principal disease-Disease code'.

All medical documentation was prepared in Polish. For the purpose of making the project publicly available, all column names have been translated into English.

**The diagnoses in the last column are according to ICD-10 -International Statistical Classification of Diseases and Health Problems.**

For example, the most common codes in the data set and the name of the corresponding disease entity:
 - **I25** - Chronic Ischemic Heart Disease,
 - **I50** - Heart failure,
 - **I21** - Acute myocardial infarction,
 - **I20** - Unstable angina.

## Project workflow

The different steps that are carried out in this project are shown in the diagram below.

<p align="center">
  <img 
    width="800"
    height="500"
    src="https://private-user-images.githubusercontent.com/81253533/290668005-d2930765-c371-4425-ac00-57a2c998238e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDI1OTE5MTIsIm5iZiI6MTcwMjU5MTYxMiwicGF0aCI6Ii84MTI1MzUzMy8yOTA2NjgwMDUtZDI5MzA3NjUtYzM3MS00NDI1LWFjMDAtNTdhMmM5OTgyMzhlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjE0VDIyMDY1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFkNDAzYzk5MTFjZGNkZTBlMjgzNzMxOWVmMGJlZWUzZjMwMWQxNzU4ZDQxYThjYjU5YTJlYTk3NjNkZDM4NjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1pQniwJDERhdJP32FMrLGZj-FnIanACl6mpgf4pDQ6o"
  >
</p>
