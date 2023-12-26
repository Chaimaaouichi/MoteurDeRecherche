from utils import calcul_similarité_texture, calcul_similarité_histogramme,Local_Descriptors_Search
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import random
import re 
import numpy as np 
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QGridLayout, QLabel, \
    QLineEdit, QMessageBox, QDialog, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImageReader
from PyQt5.QtCore import Qt
import os

#change this path the the folder where the cloned github repository is located
base_path='C://Users//Admin//Desktop//New folder//moteur-de-recherche//'

class ImageGridApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Create QLabel and QLineEdit widgets for entering values of k
        k_label = QLabel('Enter k:', self)
        self.k_edit = QLineEdit(self)

        # Create QLabel and QLineEdit widgets for entering values of l
        l_label = QLabel('Enter l:', self)
        self.l_edit = QLineEdit(self)

        # Create QPushButton for triggering the image grid creation
        btn = QPushButton('Generate Image Grid', self)
        btn.clicked.connect(self.generateImageGrid)

        # Create a QGridLayout for displaying images
        self.image_grid_layout = QGridLayout()

        # Create a QVBoxLayout to arrange the widgets vertically
        vbox = QVBoxLayout()
        vbox.addWidget(k_label)
        vbox.addWidget(self.k_edit)
        vbox.addWidget(l_label)
        vbox.addWidget(self.l_edit)
        vbox.addWidget(btn)
        vbox.addLayout(self.image_grid_layout)

        self.setLayout(vbox)
        self.setWindowTitle('Image Grid App')
        self.setGeometry(100, 100, 800, 600)  # Set initial window size

    def generateImageGrid(self):
        # Clear previous images
        for i in reversed(range(self.image_grid_layout.count())):
            self.image_grid_layout.itemAt(i).widget().setParent(None)

        try:
            # Get values of k and l from QLineEdit widgets
            k = int(self.k_edit.text())
            l = int(self.l_edit.text())

            # Get a list of image file paths from your image database

            image_database_path = base_path+'BE'
            image_file_paths = [os.path.join(image_database_path, filename) for filename in os.listdir(image_database_path)
                                if filename.endswith(('.jpg', '.png', '.jpeg'))]

            # Shuffle the list to get random images
            random.shuffle(image_file_paths)

            # Display images in the grid
            for i in range(k):
                for j in range(l):
                    if i * l + j < len(image_file_paths):
                        image_path = image_file_paths[i * l + j]

                        pixmap = QPixmap(image_path)
                        label = ClickableImageLabel(self, image_path)
                        label.setPixmap(pixmap)
                        label.setAlignment(Qt.AlignCenter)  # Center-align the image
                        self.image_grid_layout.addWidget(label, i, j)

        except ValueError:
            # Handle the case where non-integer values are entered for k and l
            QMessageBox.warning(self, 'Input Error', 'Please enter valid integer values for k and l.')
class ImageWindow(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()

        self.initUI(image_paths)

    def initUI(self, image_paths):
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        for image_path in image_paths:
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)  # Center-align the image
            layout.addWidget(label)

        self.setCentralWidget(central_widget)

        self.setWindowTitle('Image Viewer')
        self.setGeometry(100, 100, 800, 600)  # Set window size

class ClickableImageLabel(QLabel):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.PointingHandCursor)  # Set cursor to pointing hand

    def mousePressEvent(self, event):
        # Handle the mouse click event
        self.showSearchOptions()

    def showSearchOptions(self):
        dialog = SearchOptionsDialog(self,image_path=self.image_path)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            top3 = dialog.getSelectedSearchType()
            self.showImageForSearch(top3)

    def showImageForSearch(self, top3):
        # Let the user choose an image for the selected search type
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        # Iterate through image paths and display each image
        for i, image_path in enumerate(top3):
            img = mpimg.imread(image_path)
            axs[i].imshow(img)
            axs[i].axis('off')  # Turn off axis labels
            axs[i].set_title(f'Image {i+1}')
            plt.show()

            
class SearchOptionsDialog(QDialog):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.image_path=image_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Choose Search Method')

        # Create buttons for "texture search" and "histogram search"
        self.texture_button = QPushButton('Texture Search', self)
        self.histogram_button = QPushButton('Histogram Search', self)
        self.LocalDescriptor_button =QPushButton('Local Descriptors Search', self)

        # Connect button clicks to accept or reject the dialog
        self.texture_button.clicked.connect(lambda: self.acceptSearchType('Texture Search'))
        self.histogram_button.clicked.connect(lambda: self.acceptSearchType('Histogram Search'))
        self.LocalDescriptor_button.clicked.connect(lambda: self.acceptSearchType('Local Descriptors Search'))
        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.texture_button)
        layout.addWidget(self.histogram_button)
        layout.addWidget(self.LocalDescriptor_button)

        self.setLayout(layout)
    def acceptSearchType(self, search_type):
        self.selected_search_type = search_type
        self.accept()
    def getSelectedSearchType(self,):
        def replace_path(original_path):
            # Specify the part to be replaced and the replacement
            old_part = '/content/drive/MyDrive/images/BE/'
            new_part = base_path+'BE\\' 

            # Use the replace method to substitute the old part with the new part
            new_path = original_path.replace(old_part, new_part)
            return new_path
        # Return the selected search type based on which button was clicked
        if self.selected_search_type == 'Texture Search':
            image_path = self.image_path

            top3 = calcul_similarité_texture(image_path, base_path+"img_descriptors.json")
            
            top3 = [replace_path(image_name) for _, image_name in top3]
            return top3
        elif self.selected_search_type == 'Histogram Search':
            image_path = self.image_path

            top3 = calcul_similarité_histogramme(image_path, base_path+"img_indices.json")
            
            top3 = [replace_path(image_name) for _, image_name in top3]
            return top3
        elif self.selected_search_type == 'Local Descriptors Search':
            image_path=self.image_path
            S=Local_Descriptors_Search(image_path)
            imagename = os.path.basename(image_path)
            match = re.search(r'\d+', imagename)
            if match:
                 image_position = int(match.group())
            similar_indices = np.argsort(S[image_position, :])[::-1] 
            top3= similar_indices[1:4]
            top3_imagenames = [f"{index:02d}.jpg" for index in top3]
            return top3_imagenames
        else:
            return 'Unknown'


class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.initUI(image_path)

    def initUI(self, image_path):
        # Create a QLabel to display the image
        label = QLabel(self)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)  # Center-align the image
        self.setCentralWidget(label)

        self.setWindowTitle('Image Viewer')
        self.setGeometry(100, 100, 600, 400)  # Set window size

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageGridApp()
    ex.show()
    sys.exit(app.exec_())
