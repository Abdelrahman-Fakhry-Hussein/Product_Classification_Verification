
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# know if the image is readable or not
import os
import cv2
import imgaug.augmenters as iaa

def is_image_readable(image_path):
    try:
        # Attempt to read the image
        img = cv2.imread(image_path)

        if img is None:
            return False
        return True
    except Exception as e:
        print(f"Error reading image: {image_path}")
        return False


def getFiles(path=""):
    imlist = {}
    count = 0

    # Define augmentation sequence
    # if path.__contains__("Training Data"):

    augmentation = iaa.Sequential([
        # iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
        iaa.Affine(rotate=(-10, 10)),  # Rotate images by -10 to +10 degrees
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur with sigma between 0 and 1.0
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add Gaussian noise with scale between 0 and 0.05*255
        iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode='edge', keep_size=True)  # Crop and pad without changing size
    ], random_order=True)

    for each in os.listdir(path):
        imlist[each] = []
        for imagefile in os.listdir(path + '/' + each):
            image_path = path + '/' + each + '/' + imagefile
            if is_image_readable(image_path):
                # Read the image using OpenCV
                im = cv2.imread(image_path)

                # Convert BGR to RGB
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (128, 128))

                # Resize the image

                # Apply data augmentation
                if path.__contains__("Training Data"):

                    # Resize and augment
                    # Resize and convert to NumPy
                    # im = np.array(im)

                    # Augment and convert

                    augmented_images = augmentation(images=[im] * 2)  # Generate 1 augmented images per original image
                    for augmented_im in augmented_images:
                        # Convert PIL Image to numpy array
                        # augmented_im = np.array(augmented_im)
                        # augmented_im = Image.fromarray(augmented_im)
                        imlist[each].append(augmented_im)
                        count += 1
                    imlist[each].append(im)
                    count += 1

                else:
                    # im = cv2.resize(im,(128, 128))

                    imlist[each].append(im)
                    count += 1
            else:
                continue

    return [imlist, count]


# preview any image
def preview_image(imlist, key, index):
    # Get the list of images for the specified key
    image_list = imlist.get(key, [])
    if len(image_list) == 0:
        print(f"No images found for key: {key}")
        return
    if index < 0 or index >= len(image_list):
        print(f"Invalid index: {index}")
        return
    image = image_list[index]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()


import random


def split_and_shuffle_maps(map1, map2, test_size=0.2, random_state=42):
    # Combine the data from both maps
    data_combined = []
    labels_combined = []

    for key, values in map1.items():
        data_combined.extend(values)
        labels_combined.extend([key] * len(values))

    for key, values in map2.items():
        data_combined.extend(values)
        labels_combined.extend([key] * len(values))

    # Split the combined data into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(
        data_combined, labels_combined, test_size=test_size, random_state=random_state
    )

    # Shuffle the training and testing sets
    data_train, labels_train = shuffle_data(data_train, labels_train, random_state)
    data_test, labels_test = shuffle_data(data_test, labels_test, random_state)

    # Create dictionaries to store the shuffled data
    map_train = {}
    map_test = {}

    # Populate the dictionaries with shuffled data
    for label in set(labels_train):
        map_train[label] = []

    for label in set(labels_test):
        map_test[label] = []

    for data, label in zip(data_train, labels_train):
        map_train[label].append(data)

    for data, label in zip(data_test, labels_test):
        map_test[label].append(data)

    return map_train, map_test


def shuffle_data(data, labels, random_state=42):
    # Shuffle data using a random seed
    combined = list(zip(data, labels))
    random.seed(random_state)
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels
