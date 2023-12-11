import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend, layers, metrics
from keras.optimizers import Adam
from keras.applications import Xception
from keras.models import Model, Sequential
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.applications.inception_v3 import preprocess_input
import Helpers as helper

imgTraininglist, Trainingcount = helper.getFiles("E:\Data\Data\Product Recoginition\Training Data")
imgValidationlist, validationCount = helper.getFiles("E:\Data\Data\Product Recoginition\Validation Data")
print(f"train data = {Trainingcount}")
print(f"test data = {validationCount}")
def create_triplets(data_map):
    triplets = []
    for label in data_map.keys():
        anchor_data = data_map[label]

        for anchor in anchor_data:
            # Select positive example from the same label
            positive_candidates = [data for data_label, data in data_map.items() if data_label == label]
            positive = random.choice(positive_candidates)[0]

            # Select negative example from a different label
            negative_candidates = [data for data_label, data in data_map.items() if data_label != label]
            negative = random.choice(negative_candidates)[0]

            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets

train_triplets = create_triplets(imgTraininglist)
test_triplets = create_triplets(imgValidationlist)
import numpy as np

def get_batch(triplet_list, batch_size, preprocess=True):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor_batch = []
        positive_batch = []
        negative_batch = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            anchor, positive, negative = triplet_list[j]
            anchor_batch.append(anchor)
            positive_batch.append(positive)
            negative_batch.append(negative)
            j += 1

        anchor_batch = np.array(anchor_batch)
        positive_batch = np.array(positive_batch)
        negative_batch = np.array(negative_batch)

        if preprocess:
            anchor_batch = preprocess_input(anchor_batch)
            positive_batch = preprocess_input(positive_batch)
            negative_batch = preprocess_input(negative_batch)

        yield (anchor_batch, positive_batch, negative_batch)


batch_size = 20
train_generator = get_batch(train_triplets, batch_size)
test_generator = get_batch(test_triplets, batch_size)

num_plots = 6

f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

for x in get_batch(train_triplets, batch_size=num_plots, preprocess=False):
    a,p,n = x
    for i in range(num_plots):
        axes[i, 0].imshow(a[i])
        axes[i, 1].imshow(p[i])
        axes[i, 2].imshow(n[i])
        i+=1
    break
plt.show()
def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model


class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network


siamese_network = get_siamese_network()
siamese_network.summary()


class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]
siamese_model = SiameseModel(siamese_network)
optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)


def test_on_triplets(batch_size=20):
    pos_scores, neg_scores = [], []

    for data in get_batch(test_triplets, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return (accuracy, ap_mean, an_mean, ap_stds, an_stds)

save_all = False
epochs = 10
batch_size = 20

max_acc = 0
train_loss = []
test_metrics = []

for epoch in range(1, epochs + 1):
    t = time.time()

    # Training the model on train data
    epoch_loss = []
    for data in get_batch(train_triplets,batch_size=20,preprocess=False):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = test_on_triplets(batch_size=20)
    test_metrics.append(metric)
    accuracy = metric[0]

    # Saving the model weights
    if save_all or accuracy >= max_acc:
        siamese_model.save_weights("siamese_model")
        max_acc = accuracy

# Saving the model after all epochs run
siamese_model.save_weights("siamese_model-final")
def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()
def classify_images(face_list1, face_list2, threshold=0.7):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction
def ModelMetrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)*100}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()


pos_list = np.array([])
neg_list = np.array([])

for data in get_batch(test_triplets, batch_size=20):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))
    neg_list = np.append(neg_list, classify_images(a, n))
    break

ModelMetrics(pos_list, neg_list)