import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2
import joblib

import tensorflow as tf

from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn import svm
from keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Reshape, Conv2DTranspose
from keras import backend as K

class PrepaData() :
    """ Manage data preparation.

    Parameters
    ------------

    train_generator : generator
        Generator of train data for data augmentation (images databases)

    test_generator : generator
        Generator of train data for data augmentation (images databases)

    val_generator : generator
        Generator of train data for data augmentation (images databases)

    class_mode : string
        String corresponding to the type of problem for image classification
        ex : 'binary'

    img_shape : array, shape = (width, height, nb_canal)
        Array corresponding to image shape
        nb_canal = 1 if gray image, if rgb image nb_canal = 3

    classes : dict
        Dictionnary with different problem classes

    batch_size : int
        Size of batch used during training

    train_dir : string
        Path to access train data

    test_dir : string
        Path to acces test data

    val_dir : string
        Path to acces validation data

    X : DataFrame
        DataFrame contenant les données

    X_normalized : DataFrame
        DataFrame contenant les données normalisées

    y : DataFrame
        Dataframe contenant les labels

    X_train : Dataframe
        DataFrame contenant les données d'entrainement

    X_test : Dataframe
        DataFrame contenant les données de test

    X_val : Dataframe
        DataFrame contenant les données de validation

    y_train : Dataframe
        DataFrame contenant les labels d'entrainement

    y_test : Dataframe
        DataFrame contenant les labels de test

    y_val : Dataframe
        DataFrame contenant les labels de validation

    pred : array
        Array representing CNN predictions to feed ML model

    """

    def __init__(self, class_mode, X=None, y=None, batch_size = None, img_shape = None, train_dir = None, test_dir = None, val_dir = None) :
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        self.pred = None
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.classes = {}
        self.X_df = pd.DataFrame()
        self.X = X
        self.X_normalized = None
        self.y = y
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    ########################
    ########################
    ########################
    ########################
    ########################
    ## Usefull functions ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def separation_data(self, pourcentage_test = 20, choice = 1) :
        """Separate data into training and test set.

        Parameters
        ------------
        pourcentage_test : int Percentage for test set.

        """
        if choice == 1 :
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.asarray(self.X), np.asarray(self.y), test_size=pourcentage_test/100, random_state=42)

        elif choice == 2 :
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.asarray(self.pred), np.asarray(self.y), test_size=pourcentage_test/100, random_state=42)
        #self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_test, self.y_test, test_size=0.1, random_state=42)
        return None

    def normalize_data(self) :
        """Normalize data to use them to feed neural network"""
        self.X = preprocessing.normalize(self.X)
        return None

    def PCA(self) :
        pca = PCA(n_components=2)
        pca.fit(self.X)

    ########################
    ########################
    ########################
    ########################
    ########################
    ### Machine Learning ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def load_data(self, preprocess = None, reshape = 'yes') :
        """Load data into X and y"""
        x = []
        y = []
        classes = []

        for i in os.listdir(self.train_dir) :
            classes.append(i)
        train_path = []
        test_path = []
        val_path = []

        for i in range(len(classes)) :
            self.classes[classes[i]] = i

        if self.val_dir :

            for i in classes :
                chemin = self.train_dir + '/' + i
                train_path.append(chemin)
                chemin = self.test_dir + '/' + i
                test_path.append(chemin)
                chemin = self.val_dir + '/' + i
                val_path.append(chemin)

            for i in range(len(train_path)) :
                l_x = []
                for j in os.listdir(test_path[i]) :
                    chemin = test_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(train_path[i]) :
                    chemin = train_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(val_path[i]) :
                    chemin = val_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)

                for j in range(len(l_x)) :
                    img = cv2.imread(l_x[j])
                    if preprocess == 'gray' :
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    elif preprocess == 'invert' :
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = cv2.bitwise_not(img) # On inverse les couleurs
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    elif reshape :
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    x.append(img) # car dans le cas de notre problème, on ne veut que des images en niveau de gris
                for j in range(len(l_x)) :
                    y.append(i)

            self.X = x
            self.y = np.asarray(y)

        else :
            for i in classes :
                chemin = self.train_dir + '/' + i
                train_path.append(chemin)
                chemin = self.test_dir + '/' + i
                test_path.append(chemin)

            for i in range(len(train_path)) :
                l_x = []
                for j in os.listdir(test_path[i]) :
                    chemin = test_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)
                for j in os.listdir(train_path[i]) :
                    chemin = train_path[i] + '/' + j
                    chemin = chemin.replace('\\', '/')
                    l_x.append(chemin)

                for j in range(len(l_x)) :
                    img = cv2.imread(l_x[j])
                    if preprocess == 'gray' :
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    if preprocess == 'invert' :
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = cv2.bitwise_not(img) # On inverse les couleurs
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    if reshape :
                        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
                        img = np.array([cv2.resize(img,(img_width, img_height))])
                    x.append(img) # car dans le cas de notre problème, on ne veut que des images en niveau de gris
                for j in range(len(l_x)) :
                    y.append(i)

            self.X = x
            self.y = np.asarray(y)

        return None

    ########################
    ########################
    ########################
    ########################
    ########################
    ##### Deep Learning ####
    ########################
    ########################
    ########################
    ########################
    ########################

    def generateur_data(self, color_mode = 'rgb') :
        """Data augmentation for images data set. Permit to create image generators.

        Parameter
        ------------
        color_mode : str Type of images (rgb, gray, bgr, ...)
        """

        img_width, img_height = self.img_shape[0], self.img_shape[1]

        test_datagen = ImageDataGenerator(rescale=1. / 255, data_format='channels_first')

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            color_mode=color_mode,
            target_size=(img_width, img_height),
            batch_size=self.batch_size,
            class_mode=self.class_mode)

        if self.val_dir :
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                data_format='channels_first',
                horizontal_flip=True)

            self.val_generator = test_datagen.flow_from_directory(
                self.val_dir,
                target_size=(img_width, img_height),
                batch_size=self.batch_size,
                color_mode=color_mode,
                class_mode=self.class_mode)

            self.train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(img_width, img_height),
                color_mode=color_mode,
                batch_size=self.batch_size,
                class_mode=self.class_mode)

        else :
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                data_format='channels_first',
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2)

            self.train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(img_width, img_height),
                batch_size=self.batch_size,
                class_mode=self.class_mode,
                subset= 'training')

            self.val_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(img_width, img_height),
                batch_size=self.batch_size,
                class_mode=self.class_mode,
                subset = 'validation')

        return None


#----------------------------------------------------------------------------------------------------# 100 -
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#--------------------------------------------OTHER  CLASS--------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#


class MachineLearningClassifier(PrepaData) :
    """ Manage data preparation.

    Parameters
    ------------

    nb_train_samples : int
        Number of samples during training

    save_ml : string
        Path where you want to save the train model

    save_dl : string
        Path where you want to save the train model

    nb_validation_samples : int
        Number of samples for the validation

    confusion : DataFrame
        Dataframe corresponding to the confusion matrix

    input_shape : array, shape = (width, height, nb_canal) ou shape = (nb_canal, width, height)

    history : ???
        Informations about model training

    generator : model
        Generator model for GAN

    discriminator : model
        Discrominator model for GAN

    GAN : model
        GAN model

    """

    def __init__(self, epochs, nb_train_samples, nb_validation_samples, class_mode, save_ml = None, save_dl = None, X=None, y=None, batch_size = None, img_shape = None, train_dir = None, test_dir = None, val_dir = None) :
        super().__init__(class_mode, X, y, batch_size, img_shape, train_dir, test_dir, val_dir)
        self.model = None
        self.epochs = epochs
        self.save_ml = save_ml
        self.save_dl = save_dl
        self.confusion = None
        self.model_ml = None
        self.nb_validation_samples = nb_validation_samples
        self.nb_train_samples = nb_train_samples
        self.history = None
        self.input_shape = None
        self.generator = None
        self.discriminator = None
        self.GAN = None

    ########################
    ########################
    ########################
    ########################
    ########################
    ## Usefull functions ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def load_model_ml(self, nom) :
        """"Permit to load an existent model into our class object

        Parameter
        ------------
        nom : str Path to model place

        """
        self.model_ml = joblib.load(nom)


    def load_model_dl(self, nom) :
        """"Permit to load an existent model into our class object

        Parameter
        ------------
        nom : str Path to model place

        """

        self.model_dl = keras.models.load_model(nom)

    def save_model_ml(self) :
        """Save the model in order to use it later"""
        joblib.dump(self.model_ml, self.save_ml)

    def save_model_dl(self) :
        """Save the model in order to use it later"""
        self.model.save(self.save_dl)

    ########################
    ########################
    ########################
    ########################
    ########################
    ### Machine Learning ###
    ########################
    ########################
    ########################
    ########################
    ########################

    def randomForest(self) :
        """Initialize model as a default randomForest"""
        self.model_ml = RandomForestClassifier()
        return None

    def gradientBoosting(self) :
        """Initialize model as a default GradientBoostingClassifier"""
        self.model_ml = GradientBoostingClassifier()
        return None

    # Tester SVM, regressionlogistic, naive bayes, KNN
    # Sert à rien de tester le bagging avec des RandomForestClassifier
    # car c'est une méthode qui effectue déjà du bagigng

    # Out of bag randomforest

    def bagging(self) :
        """Use of bag of DecisionTreeClassifier"""
        self.model_ml = BaggingClassifier(svm.SVC(), max_samples=0.5, max_features=0.5)
        return None

    def boosting(self) :
        self.model_ml = GradientBoostingClassifier()
        return None

    def fit_ml(self, choice = 1) :
        """Fit machine learning models"""
        if choice == 1 :
            self.X_train = np.array(np.concatenate(self.X_train))
            self.X_test = np.array(np.concatenate(self.X_test))

            try :
                nsamples, nx, ny = self.X_train.shape
                self.X_train = self.X_train.reshape((nsamples,nx*ny))

                nsamples, nx, ny = self.X_test.shape # rajouter nombre de canaux pour le cas où on n'est pas en niveaux de gris
                self.X_test = self.X_test.reshape((nsamples,nx*ny))

                self.model_ml.fit(self.X_train, self.y_train)
            except :
                print('Erreur, voir fonction fit_ml')
                #gérer le cas où on n'est pas en niveau de gris
            return None
        
        else : # Dans le cas d'un entrainement sur la prédiction d'un réseau de neurones
            self.model.fit(self.X_train, self.y_train)

    def interpretation_model_ml(self) :
            """Print the confusion matrix. Usefull to interpret unbalanced problems for example."""
            y_true = self.y_test

            y_pred = (self.model_ml.predict(self.X_test))
            print('\shape : ', self.X_test.shape)
            y_actu = pd.Series(y_true.tolist(), name='Actual')
            y_predd = pd.Series(y_pred.flatten().tolist(), name='Predicted')
            confusion = pd.crosstab(y_actu, y_predd)
            print('\n')
            print('Différentes classes et leur indice : \n')
            print(self.classes)
            print('\n')
            #print('Précision du modèle : {0:.2f} %'.format(self.model.evaluate(self.X_test)[1] * 100))
            #print('\n')
            print('Matrice de confusion : \n')
            print(confusion)
            print('\n')

            print('Classification report')
            print()
            labels = ['NORMAL', 'PNEUMONIA']
            print(classification_report(y_pred, y_true, target_names=labels))

            print('coefficient kappa : ', round(cohen_kappa_score(y_true, y_pred), 2))

            print('accuracy : ', round(accuracy_score(y_true, y_pred), 2))

            accuracy  = accuracy_score(y_pred, y_true)
            precision  = precision_score(y_pred, y_true)
            recall  = recall_score(y_pred, y_true)
            f1  = f1_score(y_pred, y_true)
            roc  = roc_auc_score(y_pred, y_true)

            value = [accuracy, precision, recall, f1, roc]
            labels = ['Accuarcy', 'Precision', 'Recall', 'F1', 'ROC Score']

            plt.figure(figsize = (6, 4), dpi=150)
            plt.bar(labels, value)
            plt.title('Performance Comparision')
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.show()

            return None

    ########################
    ########################
    ########################
    ########################
    ########################
    ##### Deep Learning ####
    ########################
    ########################
    ########################
    ########################
    ########################

    def give_model(self, model) : # NE FONCTIONNE PAS, FONCTION A REFAIRE
        """With this function, user can give a model to the MachineLearningClassifier object

        Parameter
        ------------
        model : machine learning or deep learning model
        """
        self.model = model

    def cnn(self) :
        """Initialize model as a convolutional neural network"""

        # K.set_image_data_format('channels_first')
        # img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]

        # if K.image_data_format() == 'channels_first':
        #     self.input_shape = (nb_canaux, img_width, img_height)
        # else:
        #     self.input_shape = (img_width, img_height, nb_canaux)

        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        self.input_shape = (nb_canaux, img_width, img_height)

        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), input_shape=self.input_shape, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def compile(self, metrique) :
        if metrique == 'precision' :
            metric = tf.keras.metrics.Precision()
        if metrique == 'recall' :
            metric = tf.keras.metrics.Recall()
        elif metrique == 'accuracy' :
            metric = 'accuracy'
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              run_eagerly=True,
              metrics=[metric])
        return None

    def fit_gen(self) :
        """Fit model with images generator

        Returns
        -----------
        history : ?? Correspond to the history of model training
        """

        if self.val_generator :
            self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.nb_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_generator,
            )

        else :
            self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.nb_train_samples // self.batch_size,
            epochs=self.epochs,
            )

        return None

    def print_history(self, metrique) :
        """Print historic of the model training

        Parameter
        ------------
        metrique : str Name of the metric use during training
        """
        if metrique == 'precision' :
            metric = 'precision'
            metric_2 = 'val_precision'
        elif metrique == 'recall' :
            metric = 'recall'
            metric_2 = 'val_recall'
        elif metrique == 'accuracy' :
            metric = 'accuracy'
            metric_2 = 'val_accuracy'
        acc = self.history.history[metric]
        val_acc = self.history.history[metric_2]
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        if metrique == 'accuracy' :
            plt.plot(epochs, acc, 'b', label='Training acc')
            plt.plot(epochs, val_acc, 'm', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'm', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.show()

        elif metrique == 'precision' :
            plt.plot(epochs, acc, 'b', label='Training precision')
            plt.plot(epochs, val_acc, 'm', label='Validation precision')
            plt.title('Training and validation precision')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'm', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.show()

        elif metrique == 'recall' :
            plt.plot(epochs, acc, 'b', label='Training recall')
            plt.plot(epochs, val_acc, 'm', label='Validation recall')
            plt.title('Training and validation recall')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'm', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.show()

        return None

    def interpretation_model(self) :
        """Print the confusion matrix. Usefull to interpret unbalanced problems for example."""
        y_true = self.test_generator.classes

        filenames = self.test_generator.filenames
        nb_samples = len(filenames)

        y_pred = (self.model.predict(self.test_generator,steps = nb_samples) > 0.5).astype("int32")
        y_actu = pd.Series(y_true.tolist(), name='Actual')
        y_predd = pd.Series(y_pred.flatten().tolist(), name='Predicted')
        confusion = pd.crosstab(y_actu, y_predd)
        print('\n')
        print('Différentes classes et leur indice : \n')
        print(self.test_generator.class_indices)
        print('\n')
        print('Précision du modèle : {0:.2f} %'.format(self.model.evaluate(self.test_generator)[1] * 100))
        print('\n')
        print('Matrice de confusion : \n')
        print(confusion)
        print('\n')

        print('Classification report')
        print()
        labels = ['NORMAL', 'PNEUMONIA']
        print(classification_report(y_pred, y_true, target_names=labels))

        print('coefficient kappa : ', round(cohen_kappa_score(y_true, y_pred), 2))

        accuracy  = accuracy_score(y_pred, y_true)
        precision  = precision_score(y_pred, y_true)
        recall  = recall_score(y_pred, y_true)
        f1  = f1_score(y_pred, y_true)
        roc  = roc_auc_score(y_pred, y_true)

        value = [accuracy, precision, recall, f1, roc]
        labels = ['Accuarcy', 'Precision', 'Recall', 'F1', 'ROC Score']

        plt.figure(figsize = (6, 4), dpi=150)
        plt.bar(labels, value)
        plt.title('Performance Comparision')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()

        return None

    ########################
    ########################
    ########################
    ########################
    ########################
    ####### ML + DL ########
    ########################
    ########################
    ########################
    ########################
    ########################

    def dl_et_ml(self, couches_a_enlever) :
        #model = keras.models.load_model("models\cnn_preci_150x150x1.h5")

        for i in range(couches_a_enlever) :
            self.model.pop()
        self.load_data(preprocess='gray')
        self.X = np.array(self.X)
        print(self.X.shape)
        print('load ok')
        self.pred = self.model.predict(self.X)
        print(self.pred)
        print('predictions ok')
        self.separation_data(choice = 2)
        print(self.X_train)
        print(self.X_test)
        print(self.y_train)
        print(self.y_test)
        self.model_ml.fit(self.X_train, self.y_train)
        print('model entrainé')
        self.interpretation_model_ml()