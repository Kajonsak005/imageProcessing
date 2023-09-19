import os
import cv2
import numpy as np
from keras import Model, Input
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import glob
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define directories and file patterns
image_folder_path = '.face_mini/'
image_file_pattern = '**/*.jpg'
image_file_paths = glob.glob(
    f'{image_folder_path}{image_file_pattern}', recursive=True)

# Load and preprocess images
preprocessed_images = []
for file_path in image_file_paths:
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(
        image, (80, 80), interpolation=cv2.INTER_NEAREST)
    preprocessed_images.append(resized_image)

image_data = np.array(preprocessed_images)
normalized_images = image_data / 255.0

# Split the dataset
train_data, test_data = train_test_split(
    normalized_images, test_size=0.3, random_state=42)
train_data, validation_data = train_test_split(
    train_data, test_size=0.2, random_state=42)

# Add noise to the data
noise_mean = 0
noise_std_dev = 0.3
noise_factor = 0.6

train_data_noisy = train_data + noise_factor * \
    np.random.normal(loc=noise_mean, scale=noise_std_dev,
                     size=train_data.shape)
validation_data_noisy = validation_data + noise_factor * \
    np.random.normal(loc=noise_mean, scale=noise_std_dev,
                     size=validation_data.shape)
test_data_noisy = test_data + noise_factor * \
    np.random.normal(loc=noise_mean, scale=noise_std_dev, size=test_data.shape)

# Clip the data to be between 0 and 1
train_data_noisy = np.clip(train_data_noisy, 0., 1.)
validation_data_noisy = np.clip(validation_data_noisy, 0., 1.)
test_data_noisy = np.clip(test_data_noisy, 0., 1.)

# Define the autoencoder model


def create_denoising_model(optimizer='adam'):
    input_layer = Input(shape=(80, 80, 3))

    # Encoding layers
    encoded1 = Conv2D(256, (3, 3), activation='relu',
                      padding='same')(input_layer)
    encoded2 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded1)
    pooled_encoded2 = MaxPool2D((2, 2), strides=2)(encoded2)
    encoded3 = Conv2D(128, (3, 3), activation='relu',
                      padding='same')(pooled_encoded2)
    encoded4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded3)

    # Decoding layers
    decoded4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded4)
    decoded3 = Conv2D(128, (3, 3), activation='relu', padding='same')(decoded4)
    upsampled_decoded3 = UpSampling2D((2, 2))(decoded3)
    decoded2 = Conv2D(128, (3, 3), activation='relu',
                      padding='same')(upsampled_decoded3)
    decoded1 = Conv2D(256, (3, 3), activation='relu', padding='same')(decoded2)
    output_layer = Conv2D(3, (3, 3), padding='same')(decoded1)

    autoencoder_model = Model(input_layer, output_layer)
    autoencoder_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return autoencoder_model


# Create KerasRegressor wrapper
model = KerasRegressor(build_fn=create_denoising_model,
                       epochs=10, batch_size=16, verbose=1)

# Grid search epochs, batch size
param_grid = {
    'epochs': [2, 4, 8, 16],
    'batch_size': [16, 32, 64, 128],
    'optimizer': ['adam', 'sgd']
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_data_noisy, train_data)

# Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train the best model
best_model = create_denoising_model(
    optimizer=grid_result.best_params_['optimizer'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
training_history = best_model.fit(train_data_noisy, train_data,
                                  epochs=grid_result.best_params_['epochs'],
                                  batch_size=grid_result.best_params_[
                                      'batch_size'],
                                  shuffle=True,
                                  validation_data=(
                                      validation_data_noisy, validation_data),
                                  callbacks=[early_stopping],
                                  verbose=1)
best_model.save('best_denoising_autoencoder_model.h5')
