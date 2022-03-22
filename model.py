import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
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

    y_train : Dataframe
        DataFrame contenant les labels d'entrainement

    y_test : Dataframe
        DataFrame contenant les labels de test

    """

    def __init__(self, class_mode, X=None, y=None, batch_size = None, img_shape = None, train_dir = None, test_dir = None, val_dir = None) :
        self.train_generator = None
        self.test_generator = None
        self.val_generator = None
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.X = X
        self.X_normalized = None
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def normalize_data(self) :
        """Normalize data to use them to feed neural network"""

        self.X_normalized = preprocessing.normalize(self.X)
        return None

    def separation_data(self, pourcentage_test) :
        """Separate data into training and test set.

        Parameters
        ------------
        pourcentage_test : int Percentage for test set.

        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=pourcentage_test/100, random_state=42)
        return None

    def generateur_data(self) :
        """Data augmentation for images data set. Permit to create image generators."""
        
        img_width, img_height = self.img_shape[0], self.img_shape[1]

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(img_width, img_height),
            batch_size=self.batch_size,
            class_mode=self.class_mode)
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(img_width, img_height),
            batch_size=self.batch_size,
            class_mode=self.class_mode)

        if self.val_dir :
            self.val_generator = test_datagen.flow_from_directory(
                self.val_dir,
                target_size=(img_width, img_height),
                batch_size=self.batch_size,
                class_mode=self.class_mode)

        return None


#----------------------------------------------------------------------------------------------------# 100 -
#----------------------------------------------------------------------------------------------------#

#--------------------------------------------OTHER  CLASS--------------------------------------------#

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#


class MachineLearningClassifier(PrepaData) :
    """ Manage data preparation.

    Parameters
    ------------

    nb_train_samples : int
        Number of samples during training
    
    nb_validation_samples : int
        Number of samples for the validation

    confusion : DataFrame
        Dataframe corresponding to the confusion matrix
    
    input_shape : array, shape = (width, height, nb_canal) ou shape = (nb_canal, width, height)

    history : ??? 
        Informations about model training

    """

    def __init__(self, epochs, nb_train_samples, nb_validation_samples, class_mode, X=None, y=None, batch_size = None, img_shape = None, train_dir = None, test_dir = None, val_dir = None) :
        super().__init__(class_mode, X, y, batch_size, img_shape, train_dir, test_dir, val_dir)
        self.model = None
        self.epochs = epochs
        self.confusion = None
        self.nb_validation_samples = nb_validation_samples
        self.nb_train_samples = nb_train_samples
        self.history = None
        self.input_shape = None
        
    def randomForest(self) :
        """Initialize model as a default randomForest"""
        self.model = RandomForestClassifier()
        return None

    def gradientBoosting(self) :
        """Initialize model as a default GradientBoostingClassifier"""
        self.model = GradientBoostingClassifier()
        return None

    def cnn(self) :
        """Initialize model as a convolutional neural network"""
        img_width, img_height, nb_canaux = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        
        if K.image_data_format() == 'channels_first':
            self.input_shape = (nb_canaux, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, nb_canaux)
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    # FONCTION A REFAIRE CLAIREMENT
    def compile(self) :
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              run_eagerly=True,
              metrics=['accuracy'])
        return None

    def fit_gen(self) :
        """Fit model with images generator
        
        Parameters
        ------------
        train_generator : ??? Generator of images

        epochs : int    Number of epochs for the training

        validation_generator : ??? Generator of images

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

    def print_history(self) :
        """Print historic of the model training"""

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

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