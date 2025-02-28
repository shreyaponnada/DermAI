from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# Function to create and train the model
def create_and_train_model(train_dir, validation_dir, epochs=100, batch_size=32):
    # Create a base model with pre-trained weights from ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add a custom top layer for binary classification(acne(1) or non-acne(0)(basically))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Image data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Train the model
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    return model


# Function to predict whether an image contains acne
def predict_acne(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]

    prediction = model.predict(img_array)
    if prediction[0, 0] >= 0.5:
        return "Acne"
    else:
        return "Non-Acne"


# Example usage:

# Replace with the actual paths to training and test datasets
train_directory = "D:\\Acne"
validation_directory = "D:\\Acne"

# Create and train the model
trained_model = create_and_train_model(train_directory, validation_directory)

# Save the trained model
trained_model.save("acne_model.keras")

# Example of using the trained model for image recognition
test_image_path = "D:\\RobotTesting"
prediction_result = predict_acne(trained_model, test_image_path)
print(f"The image is predicted as: {prediction_result}")
