import cv2
import matplotlib.pyplot as plt

class PreTraitement() :

    def __init__(self, chemin_image):
        chemin_image = chemin_image.replace('\\', '/')
        self.image = cv2.imread(chemin_image)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_traite = self.image
        self.image_gray_traite = self.image_gray

    def egalization(self):
        self.image_gray_traite = cv2.equalizeHist(self.image_gray)
        return None

    def equal_luminosity(self):
        """
        Pour egaliser la lumière de l'image
        :return:
        """
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        self.image_traite = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        self.image_gray_traite = cv2.equalizeHist(self.image_gray)
        return None

    def show_images(self) :
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax = axes.ravel()

        ax[0].imshow(self.image_traite)
        ax[0].set_title("Image couleur")
        ax[1].imshow(self.image_gray_traite, cmap="gray")
        ax[1].set_title("Image en niveau de gris")

        fig.tight_layout()
        plt.show()

chemin_image = r"C:\Users\Utilisateur\Desktop\UQAC\Forage_de_donnees\Projet\person1_virus_6.jpeg"

"""radio = PreTraitement(chemin_image)
radio.show_images()
radio.equal_luminosity()
radio.show_images()"""