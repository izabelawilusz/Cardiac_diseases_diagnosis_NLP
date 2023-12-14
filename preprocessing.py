import pandas as pd
import filenames
import warnings

warnings.filterwarnings('ignore')


class MedicalData ():
    def __init__(self, dataframe, number_of_classes) -> None:
        self.dataframe = dataframe
        self.number_of_classes = number_of_classes
        self.columns = ['History - Onset of disease - Content',
                        'History - Physical examination - Content',
                        'Epicrisis - Physical examination - Content',
                        'Epicrisis - Medical recommendations - Content']
        self.diagnosis_columns = ["Principal disease-Disease code",
                                  "Disease code - connected"]

    def remove_empty_rows(self):
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True)

    def make_diagnosis_less_precise(self):
        self.dataframe[self.diagnosis_columns[1]] = self.dataframe[self.diagnosis_columns[0]].apply(
            lambda x: x.split(".")[0])

    def get_most_frequent_n_classes(self):
        top_classes_names = self.dataframe.iloc[:,-1].value_counts().nlargest(self.number_of_classes).index.tolist()
        return top_classes_names
    
    def remove_unnecessary_classes(self, add_class_other):
        if add_class_other:
            self.dataframe.iloc[:,-1]= self.dataframe.iloc[:,-1].apply(
                lambda x: x if x in self.get_most_frequent_n_classes() else 'other')
        else:
            self.dataframe = self.dataframe[self.dataframe.iloc[:,-1].isin(self.get_most_frequent_n_classes())]

    def class_names(self):
        return self.dataframe.iloc[:,-1].unique()
    
    def lowercase_and_remove_punctuation(self):
        for col in self.columns:
            self.dataframe[col] = self.dataframe[col].apply(
                lambda x: x.lower()
            ).apply(
                lambda x: x.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')))

    def drug_unification(self):
        counter = 0
        drug_database = pd.read_csv(filenames.medication_list_path, sep=';').astype(str)
        drug_database = pd.concat([drug_database["Name of the Medicinal Product"], drug_database['Name in common use']], axis=1)
        for col in drug_database.columns:
            drug_database[col] = drug_database[col].apply(lambda x: x.lower())
        for col in self.columns:
            for j in range(len(drug_database["Name of the Medicinal Product"])):
                medicine = drug_database["Name of the Medicinal Product"][j]
                if (str(medicine).lower() == 'nan' or medicine == "-"):
                    continue             
                medicine = str(medicine)
                for i in range(len(self.dataframe[col])):       
                    if (medicine in self.dataframe[col].iloc[i]):
                        counter +=1
                        self.dataframe[col].iloc[i] = self.dataframe[col].iloc[i].replace(medicine, 
                                                                                          drug_database["Name in common use"].iloc[j])
        print(counter)

    def remove_names(self):
        for column in self.columns:
            with open(filenames.names_path, encoding="utf-8") as f:
                names = f.read().splitlines()

            names = list(map(lambda x: x.lower(), names))
            names = list(map(lambda x: " "+x+" ", names))
            for name in names:
                for i in range(len(self.dataframe)): 
                    if str(self.dataframe[column].iloc[i]).count(name) >= 1:
                        self.dataframe[column].iloc[i]=self.dataframe[column].iloc[i].replace(name, " ")

    def remove_stopwords():
        pass

    def connect_columns(self):
        self.dataframe = pd.concat([self.dataframe["History - Onset of disease - Content"],
                                self.dataframe["History - Onset of disease - Content"] + self.dataframe["History - Physical examination - Content"],
                                self.dataframe["History - Onset of disease - Content"] + self.dataframe["History - Physical examination - Content"] 
                                + self.dataframe["Epicrisis - Physical examination - Content"],
                                self.dataframe["History - Onset of disease - Content"] + self.dataframe["History - Physical examination - Content"]
                                + self.dataframe["Epicrisis - Physical examination - Content"]+ self.dataframe["Epicrisis - Medical recommendations - Content"],
                                self.dataframe["History - Onset of disease - Content"] + self.dataframe["Epicrisis - Physical examination - Content"],
                                self.dataframe["History - Onset of disease - Content"] + self.dataframe["Epicrisis - Physical examination - Content"]
                                + self.dataframe["Epicrisis - Medical recommendations - Content"],
                                self.dataframe[self.diagnosis_columns]
                                ],
                                axis=1)
        new_columns = ["1","1_2","1_2_3","1_2_3_4", "1_3", "1_3_4"]
        self.dataframe = self.dataframe.rename(columns={k:v for (k,v) in zip([self.columns[0]] + list(range(0,5)), new_columns)})
        self.columns = new_columns
        

    def save_dataframe_classes(self, make_class_other):
        if make_class_other:
            path_to_save = filenames.data_path + "/data_{}_other.xlsx".format(self.number_of_classes)
        else:
            path_to_save = filenames.data_path + "/data_{}.xlsx".format(self.number_of_classes)
        self.dataframe.to_excel(path_to_save,  index=False)

        return path_to_save


