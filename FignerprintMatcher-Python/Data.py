import os
import random
import cv2
from matplotlib import pyplot as plt

class Dataset:
    """Class to handle fingerprint dataset operations."""
    def __init__(self, dataset_path):
        """Initialize with the path to the dataset."""
        self.dataset_path = dataset_path

    def load_file_paths(self, folder):
        """
        load the paths of every filer in the folder
        Returns:
            dict: Dictionary containing the file paths with filenames as key
        """
        item_paths = {}
        folder_path = os.path.join(self.dataset_path, folder)
        # Get all image file paths in the folder
        for item_name in os.listdir(folder_path):
            temp = item_name
            temp.replace(".", "_")
            image_attributes = tuple(temp.split("_"))
            image_path = os.path.join(folder_path, item_name)
            item_paths[image_attributes] = image_path
            # if len(item_paths) == 10: # this is just for printing and debugging purpose comment this block out!
            #     break
            # else:
            #     print(item_paths)
        return item_paths

    def load_image(self, image_path): # this is to load individual images
        return cv2.imread(image_path) # returning the images in BGR format

    def load_images(self, folder): # this function is to load all the images from a folder
        """
        load the images from a folder and return a dictionary of images
        Returns:
            dict: Dictionary containing a tuple of attributes (see load_file_paths) as keys and images as values
        Code from ChatGPT
        """
        images = {} # dictionary to store the images
        file_paths = self.load_file_paths(folder)

        for attributes, image_path in file_paths.items():
            # Load the image (using OpenCV)
            image = self.load_image(image_path)
            # Add the image to the dictionary with attributes as the key
            images[attributes] = image

        return images

    def load_bath_images(self, folder, range): # this function is to load all the images from a folder in the given range
        """
        load the images from a folder for a given range (range is a tuple with start and end points) and return a dictionary of images
        Returns:
            dict: Dictionary containing the attributes as keys and images as values
        code from Chatgpt
        """
        images = {} # dictionary to store the images
        file_paths = self.load_file_paths(folder)

        start, end = range
        # Filter file paths based on the specified range
        filtered_paths = {k: v for k, v in file_paths.items() if start <= int(k[1]) <= end}

        for attributes, image_path in filtered_paths.items():
            # Load the image (using OpenCV)
            image = self.load_image(image_path)
            # Add the image to the dictionary with attributes as the key
            images[attributes] = image

        return images



    def visualize_sample(self, folder, num_samples=9):
        """Visualize a sample fingerprint image."""
        # Load and plot the sample images
        loaded_file_paths = self.load_file_paths(folder)
        fig, axes = plt.subplots(3, 3, figsize=(15, 8))
        for i in range(num_samples):
            # print(loaded_file_paths.values())
            image_path = random.choice(list(loaded_file_paths.values()))
            print('image_path:', image_path)
            # Load the image (using OpenCV) just before visualizing
            image = cv2.imread(image_path)
            # Extract the filename from the path
            filename = os.path.basename(image_path)
            # Determine the subplot row and column
            row = i // 3
            col = i % 3
            # Visualize the image
            axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{folder}_{filename}")
        plt.show()


if __name__ == "__main__":
    # Example usage (will work if applied to the actual dataset)
    dataset = Dataset(r"C:/Users/CharB/OneDrive/Desktop/HW3_Dataset")
    file_paths = dataset.load_file_paths('Template')
    dataset.visualize_sample('Template')
    dataset.visualize_sample('TestEasy')
    dataset.visualize_sample('TestMedium')
    dataset.visualize_sample('TestHard')
