import cv2
import keras
import numpy as np
import torch
from keras.src.applications import MobileNetV2
from keras.src.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
import Helpers as helper
import tensorflow as tf
from keras.callbacks import Callback




class SaveBestModel(Callback):
    def __init__(self, filepath):
        super(SaveBestModel, self).__init__()
        self.filepath = filepath
        self.best_val_acc = -1  # Initialize with a very low value
        self.best_train_acc = -1  # Initialize with a very low value

    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')
        current_train_acc = logs.get('accuracy')
        with open('training_logs.txt', 'w') as log_file:
            log_file.write(f"Epoch {epoch + 1}/{self.params['epochs']} - "
                           f"Loss: {logs['loss']:.4f} - "
                           f"Accuracy: {logs['accuracy']:.4f} - "
                           f"Val Loss: {logs['val_loss']:.4f} - "
                           f"Val Accuracy: {logs['val_accuracy']:.4f}\n")
        if current_val_acc is not None and current_train_acc is not None:
            if current_val_acc >= self.best_val_acc and current_train_acc >= self.best_train_acc:
                self.best_val_acc = current_val_acc
                self.best_train_acc = current_train_acc
                self.model.save_weights(self.filepath, overwrite=True)
                print(f"\nModel saved with highest validation accuracy: {self.best_val_acc:.4f} and corresponding training accuracy: {self.best_train_acc:.4f}")

class CNNModelTrainer:
    def __init__(self, img_training_list,validList):
        self.image_paths = []
        self.labels = []
        self.ValImg = []
        self.valLabels = []
        self.testImg = []
        self.testLabels = []
        self.process_image_data(img_training_list,validList)

    def process_image_data(self, img_training_list,validList):
       # train_data, test_data = train_test_split(list(img_training_list.items()), test_size=test_size, random_state=42)

        # Process training data
        for label, images in img_training_list.items():
            self.image_paths.extend(images)
            self.labels.extend([label] * len(images))

        # Process test data
        for label, images in validList.items():
            self.ValImg.extend(images)
            self.valLabels.extend([label] * len(images))

    def preprocess_data(self):
        X = np.array(self.image_paths).astype('float32')
        X = X/255
        y = np.array(self.labels)
        ValX = np.array(self.ValImg).astype('float32')
        ValX = ValX/255
        Valy = np.array(self.valLabels)
        # ValX = ValX / 255
        # X = X / 255
        # label_encoder = LabelEncoder()
        # y_encoded = label_encoder.fit_transform(y)
        #
        # Valy_encoded = label_encoder.transform(Valy)
        y_encoded = [int(z)-1 for z in y]
        Valy_encoded = [int(z)-1 for z in Valy]
        Valy_encoded = np.array(Valy_encoded)
        return X, y_encoded, ValX, Valy_encoded

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(20, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, X, y_encoded, ValX, Valy_encoded, epochs=30, batch_size=16):
        # Define the ModelCheckpoint callback
        checkpoint_filepath = 'model_checkpoint.h5'
        # model_checkpoint_callback = ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_accuracy',  # You can use 'val_loss' or other metrics as well
        #     mode='max',  # 'max' if you want to save the model when the monitored quantity is maximized
        #     save_best_only=True,
        #     verbose=1
        # )
        train_datagen = ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            # horizontal_flip=True,
            # rotation_range=90
         )

        validation_datagen = ImageDataGenerator()
        save_best_model_callback = SaveBestModel(checkpoint_filepath)

        history = model.fit(
            train_datagen.flow(X, y_encoded, batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_datagen.flow(ValX, Valy_encoded, batch_size=1),
            callbacks=[save_best_model_callback]
        )

        return history

    def loadModel(self,model):
        model.load_weights('model_checkpoint.h5')
        with open('training_logs.txt', 'r') as log_file:
            logs = log_file.readlines()

        # Print the logs
        for log in logs:
            print(log.strip())
        return model

    def testM(self, listTest):
        for label, images in listTest.items():
            self.testImg.extend(images)
            self.testLabels.extend([label] * len(images))
        testX = np.array(self.testImg).astype('float32')
        testX = testX/255
        testY = np.array(self.testLabels)
        testYEncoded = [int(z) - 1 for z in testY]
        testYEncoded = np.array(testYEncoded)
        return testX ,testYEncoded
# Example usage:
img_training_list,countt = helper.getFiless("D:\cvv\Data\Product Classification","Train")  # Your image training data
imgValidationlist, validationCount = helper.getFiless("D:\cvv\Data\Product Classification","Validation")
imgtest, imgtestc = helper.getFiless("D:\cvv\Data/test","train")

#
trainer = CNNModelTrainer(img_training_list,imgValidationlist)
X, y_encoded, ValX, Valy_encoded = trainer.preprocess_data()
cnn_model = trainer.create_model()
testX,testY = trainer.testM(imgtest)
# training_history = trainer.train_model(cnn_model, X, y_encoded, ValX, Valy_encoded)
training_history = trainer.loadModel(cnn_model)
cnn_model.evaluate(testX,testY)
predictions = cnn_model.predict(testX)
#
# # Convert softmax probabilities to class labels
# # Convert softmax probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
# # Evaluate accuracy
accuracy = accuracy_score(testY, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
#
# # Generate a classification report
# print("Classification Report:")
# print(classification_report(testY, predicted_labels))
#
# # Generate a confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(testY, predicted_labels))
# # Calculate accuracy
# accuracy = np.sum(predicted_labels == testY) / len(testY)
# print(f'Accuracy: {accuracy * 100:.2f}%')