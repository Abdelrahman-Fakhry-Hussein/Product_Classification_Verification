import os
import cv2
from matplotlib import pyplot as plt
#know if the image is readable or not
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
# read images with its label and add it in a map
def getFiles(path):
    imlist = {}
    count = 0
    for each in os.listdir(path):
        imlist[each] = []
        for imagefile in os.listdir(path + '/' + each):
            image_path = path + '/' + each + '/' + imagefile
            if is_image_readable(image_path):
                im = cv2.imread(image_path)
                imlist[each].append(im)
                count += 1
            else:
                continue

    return [imlist, count]
#preview any image
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

