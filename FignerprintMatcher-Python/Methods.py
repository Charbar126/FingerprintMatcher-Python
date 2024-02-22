import cv2
import numpy as np

from Data import Dataset
class Matcher:
    """Base class for different fingerprint matching techniques."""

    def match(self, template, probe):
        """Given a template and a probe, compute a similarity score."""
        raise NotImplementedError

class MinutiaeMatching(Matcher):
    """Class for minutiae-based fingerprint matching."""
    # 1
    def binarize(self, image):
        #Sets an iamge to grayscale so the matrix has only one channel
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Threshold clarifies the pixel values assigining a maxium of 255 and min of 127 with thresh_binary making it in binary form 
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image
    
    # 2
    def skeletonize(self, image):
        """converts binary images to lines (skeletons). It takes a binary image as input and peels off
        the objects layer by layer until a skeleton remains.

        Args:
            image (matrix): a binary image

        Returns:
            matrix: This is a stripped binary image
        """
        from skimage.morphology import skeletonize  # Import skeletonize function from skimage

        skeleton_image = skeletonize(image // 255)  #binary_image // 255` is used to normalize the pixel values to either 0 or 1
        
        return skeleton_image
    
    # Helper Function
    def minutiae_at(self, pixels, i, j):
        """
        Calculate the Crossing Number (CN) of a pixel in a binary image to determine if it's a minutiae.

        Parameters:
            pixels (2D list): The binary image.
            i, j (int): The coordinates of the pixel to be checked.

        Returns:
            int: The Crossing Number (CN) of the pixel (i, j).
        """

        # Define the 8-neighborhood of pixel (i, j) in clockwise order
        neighbors = [
            pixels[i, j+1], pixels[i-1, j+1], pixels[i-1, j],
            pixels[i-1, j-1], pixels[i, j-1], pixels[i+1, j-1],
            pixels[i+1, j], pixels[i+1, j+1]
        ]

        # Initialize CN to 0
        cn = 0

        # Loop through the neighbors
        for k in range(8):
            # Compare neighbor k and neighbor k+1
            # If they are different, increment CN by 1
            # Ensure the comparison is cyclic (i.e., neighbor 8 is followed by neighbor 1)
            if neighbors[k] != neighbors[(k + 1) % 8]:
                cn += 1

        # Divide CN by 2 to finalize the calculation
        cn = cn // 2

        return cn
    
    

    def extract_minutiae(self, skeleton):
        """
        Extract minutiae (ridge ending and bifurcation) from a skeletonized fingerprint image.

        Parameters:
            skeleton (2D numpy array): The skeletonized binary image.

        Returns:
            minutiae_term (2D numpy array): Binary image showing ridge endings.
            minutiae_bif (2D numpy array): Binary image showing ridge bifurcations.
        """

        # Initialize arrays to hold the minutiae maps
        minutiae_term = np.zeros_like(skeleton, dtype=np.uint8)
        minutiae_bif = np.zeros_like(skeleton, dtype=np.uint8)

        # Iterate through each pixel in the skeleton image
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):

                # Check for endpoint
                # A pixel is an endpoint if it is a ridge pixel (skeleton[i, j] == 1)
                # and has a CN of 1
                if skeleton[i, j] == 1 and self.minutiae_at(skeleton, i, j) == 1:
                    minutiae_term[i, j] = 1

                # Check for bifurcation
                # A pixel is a bifurcation if it is a ridge pixel (skeleton[i, j] == 1)
                # and has a CN of 3
                elif skeleton[i, j] == 1 and self.minutiae_at(skeleton, i, j) == 3:
                    minutiae_bif[i, j] = 1

        return minutiae_term, minutiae_bif
   
    def match(self, template, probe):
        """Compare the minutiae points of the template and probe."""
        #Grab minutiae details from both fingerprints
        template_minutiae_term, template_minutiae_bif = self.get_minutiae_of_fingerprint(template)
        probe_minutiae_term, probe_minutiae_bif = self.get_minutiae_of_fingerprint(probe)
        
        #Threshold for match
        distance_threshold=20
        
        #Put all given minutate points into a matrix
        minutiae1 = np.vstack((template_minutiae_term, template_minutiae_bif))
        minutiae2 = np.vstack((probe_minutiae_term, probe_minutiae_bif))
        matched_pairs = []
        
        # Iterate through all minutiae in the first set
        for m1 in minutiae1:
            # Find the closest minutia in minutiae2
            min_distance = np.min(np.linalg.norm(minutiae2 - m1, axis=1))
            # If the minimum distance is below the threshold, consider it a match
            if min_distance < distance_threshold:
                matched_pairs.append((m1, minutiae2[np.argmin(np.linalg.norm(minutiae2 - m1, axis=1))]))

        # Normalize by the total number of minutiae in both sets
        match_score = len(matched_pairs) / (len(minutiae1) + len(minutiae2))
        return match_score

    def get_minutiae_of_fingerprint(self, image):
        """Converts the fingerprint from binary to skeleton and extracts minutiae points.

        Args:
            image (numpy.ndarray): Binary representation of the fingerprint image.

        Returns:
           Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing arrays representing termination and bifurcation minutiae points.
        """
        image_binary = self.binarize(image)
        image_skeleton = self.skeletonize(image_binary)
        minutiae_term, minutiae_bif = self.extract_minutiae(image_skeleton)
        return minutiae_term, minutiae_bif

class FeatureDescriptorMatching(Matcher):
    """Class for feature descriptor (e.g., SIFT/SURF) based fingerprint matching."""

    def __init__(self, method="SIFT"):
        """Initialize with the chosen descriptor method, sift model and bf matcher."""
        self.method = method
        self.sift_detector = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def extract_features(self, image):
        """Extract features using the chosen descriptor from the given fingerprint image."""
        keypoints, descriptors = self.sift_detector.detectAndCompute(image, None)
        #Keypoints not needed for bf comparrison
        return descriptors

    def match(self, template, probe):
        """Compare the features of the template and probe."""
        #Grab descriptors
        template_descriptors = self.extract_features(template)
        probe_descriptors = self.extract_features(probe)
        bf_matches = self.bf.knnMatch(template_descriptors, probe_descriptors, k=2)
        bf_good = [m for m, n in bf_matches if m.distance < 0.75 * n.distance]
        return len(bf_good) / len(bf_matches)

class PretrainedCNNModel(Matcher):
    """Class for pretrained CNN model-based fingerprint matching."""

    def __init__(self, model_name):
        """Initialize with the chosen pretrained model's name."""
        self.model_name = model_name
        self.basemodel= VGG16(weights='imagenet')
        
    def preprocess_img(self, img_path):
        """
        Load and preprocess the image.
        """
        img = image.load_img(img_path, target_size=(224, 224))  # Load and resize image
        img_array = image.img_to_array(img)  # Convert image to array
        expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
        preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess image
        return preprocessed_img
    
    def feature_extraction(self, model, preprocessed_img):
        """
        Use the CNN model to extract features from the preprocessed image.
        """
        features = model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)  # Normalize
        return normalized_features


    def match(self, template, probe):
        """Use the pretrained model to match the template and probe."""
        # Placeholder for matching logic
        return 0.0


if __name__ == "__main__":
     # Example usage for testing MinutiaeMatching class
    dataset_path = r"C:/Users/CharB/OneDrive/Desktop/HW3_Dataset"
    dataset = Dataset(dataset_path)

    # Load template and probe fingerprints
    template_images = dataset.load_images('Template')
    probe_easy_images = dataset.load_images('TestEasy')
    probe_medium_images = dataset.load_images('TestMedium')
    probe_hard_images = dataset.load_images('TestHard')
    print("datasets loaded")

    # Create MinutiaeMatching instance
    minutiae_matcher = MinutiaeMatching()
    
    #Create SIFT FeatureDesciptorMatching instance
    sift_feature_matcher = FeatureDescriptorMatching()
    
    cnn_feature_matcher = PretrainedCNNModel()

    #Testing minuate matching
    # for template_attributes, template_image in template_images.items():
    #         for probe_attributes, probe_image in probe_easy_images.items():
    #             # Call the match function
    #             similarity_score = minutiae_matcher.match(template_image, probe_image)
    #             # Print or use the similarity score as needed
    #             print(f"Similarity score between {template_attributes} and {probe_attributes}: {similarity_score}")
                
    #Testing SIFT FeatureDesciptor matching
    for template_attributes, template_image in template_images.items():
            for probe_attributes, probe_image in probe_medium_images.items():
                # Call the match function
                similarity_score = sift_feature_matcher.match(template_image, probe_image)
                # Print or use the similarity score as needed
                print(f"Similarity score between {template_attributes} and {probe_attributes}: {similarity_score}")
                
                
    #Testing CNN FeatureDesciptor matching
    for template_attributes, template_image in template_images.items():
            for probe_attributes, probe_image in probe_hard_images.items():
                # Call the match function
                similarity_score = PretrainedCNNModel.match(template_image, probe_image)
                # Print or use the similarity score as needed
                print(f"Similarity score between {template_attributes} and {probe_attributes}: {similarity_score}")
                
                
                