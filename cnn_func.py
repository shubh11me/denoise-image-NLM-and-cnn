import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image

import cv2


import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D


# Load the dataset

def cnnn(path):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Preprocess the images
# train_images = train_images.reshape((len(train_images), 784))
# test_images = test_images.reshape((len(test_images), 784))

# x_train = np.reshape(train_images, (train_images.shape[0], 784, 784))
# x_test = np.reshape(test_images, (test_images.shape[0], 784, 784))

# original_image_shape = train_images.shape[1:]

# train_images = train_images.reshape((len(train_images), *original_image_shape))
# test_images = test_images.reshape((len(test_images), *original_image_shape))


    num_images = 5
    train_images = train_images[:num_images]
    train_labels = train_labels[:num_images]

    test_images = test_images[:num_images]
    test_labels = test_labels[:num_images]

    new_shape = (256, 256)

# Create an empty array to store the reshaped images
    reshaped_images = np.empty((train_images.shape[0],) + new_shape + (3,))

# Reshape each image using interpolation
    for i, image in enumerate(train_images):
    # Convert to PIL Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert("RGB")
    # Resize the image to the new shape
        resized_image = pil_image.resize(new_shape, Image.ANTIALIAS)

    # Convert back to NumPy array
        reshaped_images[i] = np.array(resized_image)

        train_images=reshaped_images
        reshaped_images = np.empty((test_images.shape[0],) + new_shape+ (3,))

# Reshape each image using interpolation
    for i, image in enumerate(test_images):
    # Convert to PIL Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert("RGB")
    # Resize the image to the new shape
        resized_image = pil_image.resize(new_shape, Image.ANTIALIAS)

    # Convert back to NumPy array
        reshaped_images[i] = np.array(resized_image)
    test_images=reshaped_images

    print(train_images.shape)
    print(test_images.shape)
# exit()



# resized_train_images = []
# for image in train_images:
#     resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
#     resized_train_images.append(resized_image)

# train_images = tf.data.Dataset.from_tensor_slices(resized_train_images).batch(128)
# print(train_images[0][0].shape)
# for image_batch in train_images:
#     for image in image_batch:
#         print(image.shape)



# Reshape test images
# resized_test_images = []
# for image in test_images:
#     resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
#     resized_test_images.append(resized_image)
# test_images = tf.data.Dataset.from_tensor_slices(resized_test_images).batch(128)

# train_images=resized_train_images
# test_images=resized_train_images

# print(train_images.shape)
# print(test_images.shape)
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

# train_images = np.expand_dims(train_images, axis=0)
# test_images = np.expand_dims(test_images, axis=0)

    print(train_images.shape)
# Add Gaussian noise to the images
    train_images_noisy = train_images + 0.5 * tf.random.normal(shape=train_images.shape)
    test_images_noisy = test_images + 0.5 * tf.random.normal(shape=test_images.shape)


# Define the CNN architecture
    model = keras.Sequential(
        [
            layers.Input(shape=(256, 256, 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
        ]
    )
    model.summary()
# Compile the model
    model.compile(optimizer="adam", loss="mse")

# Train the model
    print("herrrrr")
    model.fit(train_images_noisy, train_images, epochs=2, batch_size=1)
    print("zall")

# Evaluate the model
    score = model.evaluate(test_images_noisy, test_images, verbose=1)
    print("Test loss:", score)

# Apply the model to denoise an image
# import cv2
# import numpy as np

# Load an image
# img = cv2.imread('color1.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('color1.jpg', cv2.IMREAD_COLOR)

# # Preprocess the image
# print("scsccssc")
# img = cv2.resize(img, (256, 256))
# img = img.astype("float32") / 255.0
# print("x")

# img = np.expand_dims(img, axis=0)
# img = np.expand_dims(img, axis=-1)
# print("xzz")


# cv2.imwrite('denoised_image.png', img[0] * 255.0)
# # Denoise the image
# denoised_img = model.predict(img)
# print("xaaaaaaaaaaaaa")

# # denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

# # Save the denoised image
# cv2.imwrite('denoised_image.png', denoised_img[0] * 255.0)


# cv2.imwrite('denoised_image.png', denoised_img)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

# Preprocess the image
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

# Denoise the image
    denoised_img = model.predict(img)

# Rescale the denoised image back to the range [0, 255]
    denoised_img = denoised_img[0] * 255.0
    denoised_img = denoised_img.astype("uint8")
# denoised_img = np.uint8(np.clip(denoised_img * 255.0, 0, 255))
    return {'img':denoised_img,'shape':train_images.shape,'score':score}
# Save the denoised image
    # cv2.imwrite('denoised_image.jpeg', denoised_img)



def denoise_image_cnn_without_model_train(image_path):
    # Load the noisy image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values
    image = image.astype('float32') / 255.0

    # Add noise to the image (optional)
    noisy_image = image + np.random.normal(loc=0, scale=0.1, size=image.shape)

    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=noisy_image.shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(np.expand_dims(noisy_image, axis=0), np.expand_dims(image, axis=0), epochs=10, batch_size=1)

    # Denoise the image
    denoised_image = model.predict(np.expand_dims(noisy_image, axis=0))

    # Remove the extra dimension
    denoised_image = np.squeeze(denoised_image)

    # Rescale the denoised image to [0, 1]
    denoised_image = (denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image))

    # Convert back to BGR
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)

    return denoised_image * 255.0



def cnn_b(path):


    def remove_gaussian_noise(image_path):
        # Load the grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply bilateral filter
        denoised_image = cv2.bilateralFilter(image, 8, 85, 85)

        return denoised_image


# Load the noisy image
# image = cv2.imread('noisy_image.jpeg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Normalize the pixel values
    image = image.astype('float32') / 255.0
    image = cv2.resize(image  , (256 , 256))

# Add noise to the image (optional)
    print(image.shape)
    noisy_image = image + np.random.normal(loc=0, scale=0.1, size=image.shape)

# Reshape the image for CNN input
    input_image = np.expand_dims(noisy_image, axis=2)

# Define the CNN model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_image.shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

# Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
    model.fit(input_image, image, epochs=10, batch_size=1)

# Denoise the image
    denoised_image = model.predict(input_image)

# Remove the extra dimension
    denoised_image = np.squeeze(denoised_image)

# Rescale the denoised image to [0, 1]
    denoised_image = (denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image))

# Adjust the contrast of the denoised image
    denoised_image = cv2.normalize(denoised_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('temp/denoised4_image.jpeg', denoised_image)

    denoised_image=remove_gaussian_noise('temp/denoised4_image.jpeg')
# Save the denoised image
    return denoised_image