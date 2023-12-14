import os
from keras.backend import clear_session
import pandas as pd
import numpy as np
import gensim
import traceback
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import f1_score
from imblearn.combine import SMOTETomek 
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from xgboost import XGBClassifier
from keras.utils import np_utils
from typing import List, Tuple



class OptunaClassification:

    def __init__(self, dataframe : pd.DataFrame, i_column : str, work_path : str, class_names : np.ndarray) -> None: 
        self.dataframe = dataframe
        self.i_column = i_column
        self.work_path = work_path
        self.class_names = class_names

    def prepare_data(self, dataframe: pd.DataFrame, input_column : str, vectorization_method : str
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = dataframe[input_column]
        y = dataframe.iloc[:,-1]
        if vectorization_method != "word2vec":
            X = X.values.reshape(-1,1)
            y = y.values
            X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size=0.2,shuffle=True)
            X_train=pd.DataFrame(X_train)[0]
            X_test=pd.DataFrame(X_test)[0]
            y_train=pd.DataFrame(y_train)[0]
            y_test=pd.DataFrame(y_test)[0]

            if vectorization_method == "tf_idf":
                vectorizer = TfidfVectorizer(use_idf=True)
            elif vectorization_method == "count_vectorizer":
                vectorizer = CountVectorizer()
            else:
                traceback.print_exc()
                raise ValueError("Invalid vectorization method. Please choose 'tf_idf' or 'count_vectorizer' or 'word2vec'.")        
            X_train_vector = vectorizer.fit_transform(X_train)
            X_test_vector = vectorizer.transform(X_test) 
        else:
            X = X.squeeze()
            y = y.squeeze()
            X =X.apply(lambda x: gensim.utils.simple_preprocess(x))
            X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size=0.2,shuffle=True)
            w2v_model = gensim.models.Word2Vec(X_train,
                                    vector_size=100,
                                    window=5,
                                    min_count=2)
            words = set(w2v_model.wv.index_to_key )
            X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in X_train], dtype="object")
            X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in X_test], dtype="object")
            X_train_vector = []
            for v in X_train_vect:
                if v.size:
                    X_train_vector.append(v.mean(axis=0))
                else:
                    X_train_vector.append(np.zeros(100, dtype=float))
                    
            X_test_vector = []
            for v in X_test_vect:
                if v.size:
                    X_test_vector.append(v.mean(axis=0))
                else:
                    X_test_vector.append(np.zeros(100, dtype=float))            
            y_train = y_train.values.ravel()
            X_train_vector = np.array(X_train_vector)
            X_test_vector = np.array(X_test_vector)

        return (X_train_vector, X_test_vector, y_train, y_test)

    def clear_dataframe (self, dataframe : pd.DataFrame, classic_or_network : str) -> pd.DataFrame:
        if classic_or_network == "network":
            dictionary = {"value": "test_accuracy", 
                          "params_resampling_method": "resampling", 
                          "user_attrs_cm" : "cm", 
                          "params_activation_function" : "activation_function", 
                          "params_dropout_level" : "dropout_level", 
                          "params_optimizer": "optimizer", 
                          "params_vectorization_method":"vectorization_method",  
                          "params_more_layers":"more_layers", 
                          "params_batch_size":"batch_size", 
                          "user_attrs_test_precision": "test_precision", 
                          "user_attrs_test_recall": "test_recall", 
                          "user_attrs_f1_score": "test_f1_score"}
        else: 
            dictionary = {"value": "test_accuracy", 
                          "params_resampling_method": "resampling", 
                          "params_classification_method": "classification_method",
                          "user_attrs_cm" : "cm",
                          "params_vectorization_method":"vectorization_method", 
                          "user_attrs_test_precision": "test_precision", 
                          "user_attrs_test_recall": "test_recall", 
                          "user_attrs_f1_score": "test_f1_score"}
        dataframe = dataframe.rename(columns = dictionary)
        cols = list(dataframe.columns.values)
        cols.pop(cols.index('datetime_start')) 
        cols.pop(cols.index('datetime_complete'))
        cols.pop(cols.index('duration'))
        dataframe = dataframe[cols+['datetime_start','datetime_complete', 'duration']]
        dataframe = dataframe.sort_values('test_accuracy', ascending= False)

        return dataframe


    def return_predicted(self, 
                        trial_name : int, 
                        X_train : pd.DataFrame, 
                        X_test : pd.DataFrame, 
                        y_train : pd.DataFrame, 
                        y_test : pd.DataFrame, 
                        resampling_method : str, 
                        classification_method : str
                        ) -> Tuple[pd.Series, pd.Series]:

        if resampling_method == "undersampling":
            under_sampler= RandomUnderSampler(sampling_strategy='auto')
            X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        if resampling_method == "oversampling":
            over_sampler= RandomOverSampler(sampling_strategy='auto')
            X_train, y_train= over_sampler.fit_resample(X_train, y_train)
        if resampling_method == "none":
            pass
        if classification_method == "logistic_reg":
            classifier = LogisticRegression(max_iter = 1000)
        if classification_method == "svc":
            classifier = svm.SVC()
        if classification_method == "random_forest":
            classifier = RandomForestClassifier()
        if classification_method == "knn":
            classifier = KNeighborsClassifier()  
        if classification_method == "xgb":
            classifier = XGBClassifier(max_depth=2, n_estimators=30) 
            label_encoder = LabelEncoder()
            target = label_encoder.fit_transform(y_train)
            classifier.fit(X_train, target)
            y_pred = classifier.predict(X_test)
            if not os.path.exists(self.work_path + '/{}'.format(self.i_column)):
                os.makedirs(self.work_path + '/{}'.format(self.i_column))
            with open(self.work_path + '/{}/model_{}.pkl'.format(self.i_column, trial_name), 'wb') as f:
                pickle.dump(classifier, f)
            y_pred_ = label_encoder.inverse_transform(y_pred)

            classes = self.class_names
            y_predicted= []
            for element in y_pred_:
                for i in classes:
                    if element == i:
                        y_predicted.append(i)

        if classification_method != "xgb":
            classifier.fit(X_train, y_train)
            if not os.path.exists(self.work_path + '//{}'.format(self.i_column)):
                os.makedirs(self.work_path + '/{}'.format(self.i_column))
            with open(self.work_path + '/{}/model_{}.pkl'.format(self.i_column, trial_name), 'wb') as f:
                pickle.dump(classifier, f)
            y_predicted = classifier.predict(X_test)

        return (y_test, y_predicted)
    
    def return_predicted_network(self, 
                                trial_name : int,
                                X_train : pd.DataFrame, 
                                X_test : pd.DataFrame, 
                                y_train : pd.DataFrame, 
                                y_test : pd.DataFrame, 
                                resampling_method : str, 
                                vectorization_method : str, 
                                activation_function : str, 
                                dropout_level : float, 
                                optimizer : str, 
                                more_layers : bool, 
                                batch_size : int
                                ) -> Tuple[pd.Series, pd.Series]:

        if resampling_method == "undersampling":
            under_sampler= RandomUnderSampler(sampling_strategy='auto')
            X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        if resampling_method == "oversampling":
            over_sampler= RandomOverSampler(sampling_strategy='auto')
            X_train, y_train= over_sampler.fit_resample(X_train, y_train)
        if resampling_method == "smote":
            X_train=X_train.toarray()
            X_test=X_test.toarray()
            X_train, y_train= SMOTETomek().fit_resample(X_train, y_train)
        if resampling_method == "none":
            pass

        y_test, y_predicted = self.neural_network(trial_name,
                                                  X_train, 
                                                  X_test, 
                                                  y_train, 
                                                  y_test, 
                                                  activation_function, 
                                                  vectorization_method, 
                                                  dropout_level, 
                                                  optimizer, 
                                                  more_layers, 
                                                  batch_size)

        return (y_test, y_predicted)

    def neural_network(self, 
                       trial_name : int, 
                       X_train : pd.DataFrame, 
                       X_test : pd.DataFrame, 
                       y_train : pd.DataFrame, 
                       y_test : pd.DataFrame, 
                       activation_function : str, 
                       vectorization_method : str, 
                       dropout_level : float, 
                       optimizer : str, 
                       more_layers : bool, 
                       batch_size : int
                       ) -> Tuple[pd.Series, List[float]]:
        
        number_of_classes = len(self.class_names)
        try:
            X_train.sort_indices()
            X_test.sort_indices()
        except:
            pass
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(y_train)
        dummy_y_train = np_utils.to_categorical(target, dtype ="uint8")
        

        def build_model():
            model = Sequential()
            model.add(Dense(256, input_dim=X_train.shape[1], activation=activation_function))
            model.add(Dropout(dropout_level))
            model.add(Dense(200, activation=activation_function))
            model.add(Dropout(dropout_level))
            model.add(Dense(160, activation=activation_function))
            model.add(Dropout(dropout_level))
            if more_layers:
                model.add(Dense(64, activation=activation_function))     
                model.add(Dropout(dropout_level)) 
                model.add(Dense(32, activation=activation_function))     
                model.add(Dropout(dropout_level)) 
                model.add(Dense(16, activation=activation_function))     
                model.add(Dropout(dropout_level))                 
            model.add(Dense(number_of_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            model.summary()
            return model

        if vectorization_method == "word2vec":
            classifier = KerasClassifier(build_fn=build_model, epochs=600, batch_size=batch_size)
        else:
            classifier = KerasClassifier(build_fn=build_model, epochs=15, batch_size=batch_size) 
        classifier.fit(X_train, dummy_y_train)
        classifier.model.save(self.work_path + '/{}/model_{}.h5'.format(self.i_column, trial_name))
        y_pred = classifier.predict(X_test)
        y_pred_ = label_encoder.inverse_transform(y_pred)

        classes=self.class_names
        y_predicted= []

        for element in y_pred_:
            for i in classes:
                if element == i:
                    y_predicted.append(i)
        
        return y_test, y_predicted

    def return_accuracy (self, y_test : pd.Series, y_predicted : List[float]) -> float:
        accuracy = accuracy_score(y_test, y_predicted)

        return accuracy

    def objective_ml(self, trial : int):
        clear_session()
        trial_name = trial.number
        vectorization_method = trial.suggest_categorical("vectorization_method", ["count_vectorizer", "tf_idf", "word2vec"])
        resampling_method = trial.suggest_categorical("resampling_method", ["oversampling", "none"])
        classification_method = trial.suggest_categorical("classification_method", ["logistic_reg", "svc", "random_forest", "knn", "xgb"])
        X_train, X_test, y_train, y_test = self.prepare_data( self.dataframe, self.i_column, vectorization_method)
        y_test, y_predicted = self.return_predicted(trial_name,X_train, X_test, y_train, y_test, resampling_method, classification_method)
        cm = confusion_matrix(y_test, y_predicted)
        trial.set_user_attr("cm", str(cm))
        test_accuracy = self.return_accuracy(y_test, y_predicted)
        f1score= f1_score(y_test, y_predicted , average =None)
        trial.set_user_attr("test_f1-score", f1score)
        recall= recall_score(y_test, y_predicted, average=None)
        trial.set_user_attr("test_recall", recall)
        precision= precision_score(y_test, y_predicted, average =None)
        trial.set_user_attr("test_precision", precision)   

        return test_accuracy

    def objective_network(self, trial : int):
        clear_session()
        trial_name = trial.number
        vectorization_method = trial.suggest_categorical("vectorization_method", ["count_vectorizer", "tf_idf", "word2vec"])
        resampling_method = trial.suggest_categorical("resampling_method", ["undersampling",  "oversampling", "none"])
        activation_function = trial.suggest_categorical("activation_function", ["relu", "elu"])
        dropout_level= trial.suggest_categorical("dropout_level", [0.5, 0.2, 0.3, 0.4])
        optimizer= trial.suggest_categorical("optimizer", ["adam", "RMSprop", "SGD"])
        more_layers = trial.suggest_categorical("more_layers", [True, False])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        X_train, X_test, y_train, y_test = self.prepare_data( self.dataframe, self.i_column, vectorization_method)
        y_test, y_predicted= self.return_predicted_network(trial_name, 
                                                           X_train, 
                                                           X_test, 
                                                           y_train, 
                                                           y_test, 
                                                           resampling_method,
                                                           vectorization_method, 
                                                           activation_function, 
                                                           dropout_level, 
                                                           optimizer, 
                                                           more_layers, 
                                                           batch_size)
        cm = confusion_matrix(y_test, y_predicted)
        trial.set_user_attr("cm", str(cm))
        test_accuracy = self.return_accuracy(y_test, y_predicted)
        f1score= f1_score(y_test, y_predicted , average =None)
        trial.set_user_attr("test_f1-score", f1score)
        recall= recall_score(y_test, y_predicted, average=None)
        trial.set_user_attr("test_recall", recall)
        precision= precision_score(y_test, y_predicted, average =None)
        trial.set_user_attr("test_precision", precision)       

        return test_accuracy