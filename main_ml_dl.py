from model import *

train_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\train"
validation_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\val"
test_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\test"

classifier = MachineLearningClassifier(save_ml = "random_forest_150x150x1_mldl.joblib",
                                        save_dl = "cnn_150x150x1_mldl.h5",
                                        epochs = 50,
                                        nb_train_samples=5212,
                                        nb_validation_samples=17,
                                        class_mode = 'binary',
                                        batch_size = 16,
                                        img_shape = (150,150,1),
                                        train_dir = train_data_dir,
                                        test_dir = test_data_dir,
                                        val_dir = validation_data_dir)

# Deep Learning
classifier.generateur_data(color_mode = 'grayscale')

classifier.cnn()
classifier.compile(metrique = 'recall')

classifier.fit_gen()
# Machine learning + entrainement CNN
classifier.randomForest()
print('\n')
classifier.dl_et_ml(couches_a_enlever=3)
print('entrainement ok')

#classifier.interpretation_model_ml()