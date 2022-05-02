from model import *

train_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\train"
validation_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\val"
test_data_dir = r"C:\Users\Utilisateur\Desktop\bdd_xray\archive\chest_xray\test"

classifier = MachineLearningClassifier(epochs = 1,
                                        nb_train_samples=5212,
                                        nb_validation_samples=17,
                                        class_mode = 'binary',
                                        batch_size = 16,
                                        img_shape = (150,150,1),
                                        train_dir = train_data_dir,
                                        test_dir = test_data_dir,
                                        val_dir = validation_data_dir)

#classifier.generateur_data(color_mode = 'grayscale')

classifier.cnn()
classifier.model.summary()
# classifier.compile(metrique = 'recall')

# classifier.fit_gen()
# classifier.print_history(metrique='recall')

# classifier.interpretation_model()
# classifier.save_model_dl("test.h5")