from model import *

train_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\train"
validation_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\val"
test_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\test"

classifier = MachineLearningClassifier(save = "random_forest_150x150x1.joblib",
                                        img_shape = (150,150,1),
                                        train_dir = train_data_dir,
                                        test_dir = test_data_dir,
                                        val_dir = validation_data_dir)
classifier.load_data(preprocess='gray')
print('load ok')
classifier.separation_data()
print('sep ok')
print('\n')
print("données d'entrainement :")
print(len(classifier.X_train))
print("données de test :")
print(len(classifier.X_test))
classifier.randomForest()
print('\n')
classifier.fit_ml()
print('entrainement ok')
classifier.interpretation_model_ml()
#classifier.save_model()