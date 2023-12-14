# Cardiac_diseases_diagnosis_NLP

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
