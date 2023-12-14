import pandas as pd
from classification import OptunaClassification
import filenames
from preprocessing import MedicalData
import optuna
from optuna.samplers import RandomSampler
import shutil
import os
import argparse


if __name__ == "__main__":
    number_of_classes = [2,3,4,5]
    with_class_other = [True, False]
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classic_or_network", type=str, default=["ml", "network"], choices=["ml", "network"], nargs='+',)
    parser.add_argument("-n", "--n_trials", default=50, type=int)
    parser.add_argument("-d", "--data_preparation", action="store_true")
    args = parser.parse_args()
    
    if args.data_preparation:
        print("DATA PREPARATION \n")
        dataframe = pd.read_excel(filenames.medical_data_path)

        for class_other_option in with_class_other:
            for n in number_of_classes:
                if class_other_option == False:
                    print("Start preparing data for {} classes".format(n))
                else:
                    print("Start preparing data for {} classes with class other".format(n))
                medical_data = MedicalData(dataframe, n)
                medical_data.remove_empty_rows()
                print("\t Empty rows were removed.")
                medical_data.make_diagnosis_less_precise()
                print("\t ICD-10 diagnoses were simplified.")
                medical_data.remove_unnecessary_classes(class_other_option)
                print("\t Unnecessary classes were removed.")
                medical_data.lowercase_and_remove_punctuation()
                print("\t Lowercasing and removing punctuation were applied.")
                medical_data.drug_unification()
                print("\t Drug unification was applied.")
                medical_data.remove_names()
                print("\t Names were removed.")
                medical_data.connect_columns()
                print("\t Connected colums were added.")
                medical_data.save_dataframe_classes(class_other_option)

    print("OPTUNA STUDIES")
    classic_or_network = args.classic_or_network
    if "ml" in classic_or_network and "network" not in classic_or_network:
        classic_or_network = ["ml"]
    elif "network" in classic_or_network and "ml" not in classic_or_network:
        classic_or_network = ["network"]

    for model_option in classic_or_network:
        for class_other_option in with_class_other:
            for n in number_of_classes:
                if class_other_option == False:
                    print("Start classification for {} classes".format(n))
                    dataframe_name = "data_{}".format(n)
                else:
                    print("Start classification for {} classes with class other".format(n))
                    dataframe_name = "data_{}_other".format(n)
                dataframe = pd.read_excel(filenames.data_path + "/" + dataframe_name + ".xlsx")
                medical_data = MedicalData(dataframe, n)
                class_names = medical_data.class_names()
                results_path = filenames.output_path + "/" + model_option + "/results"
                models_path = filenames.output_path + "/" + model_option + "/models"
                paths = [results_path, models_path]
                for n_path in paths:
                    if not os.path.exists(n_path):
                        os.makedirs(n_path)
                writer = pd.ExcelWriter((results_path+ '/results_{}.xlsx').format(dataframe_name), engine="xlsxwriter")
                input_column = ["1", "1_2", "1_2_3", "1_2_3_4", "1_3", "1_3_4"]

                for i_column in input_column:
                    print("\t Study for column {}".format(i_column))
                    models_path_column = models_path + "/" + dataframe_name + "/" + i_column
                    if os.path.exists(models_path_column):
                        shutil.rmtree(models_path_column)
                    else:
                        pass        
                    opt_class= OptunaClassification(dataframe, i_column, filenames.output_path + "/"+ model_option + "/models/"+ dataframe_name, class_names)
                    study = optuna.create_study(study_name= i_column,
                                                    direction="maximize",
                                                    sampler=RandomSampler(),
                                                    pruner = optuna.pruners.MedianPruner())
                    n_trials = args.n_trials
                    if model_option == "network":
                        study.optimize(opt_class.objective_network, n_trials=n_trials, timeout=None)
                    else:
                        study.optimize(opt_class.objective_ml, n_trials=n_trials, timeout=None)
                    best_results_df = study.trials_dataframe()
                    best_results_df = opt_class.clear_dataframe(best_results_df, model_option)
                    best_results_df.to_excel(writer, sheet_name= i_column)
                    print("Study finished for {}".format(i_column))

                writer.close()    

