# Importar librerías necesarias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import accuracy_score
from scipy import stats
import os

# Rutas y separación de la data
base_dir = './data/'
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/validation'
test_dir = f'{base_dir}/test'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Generador para los datos de validación y prueba (solo escalado)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generador de datos de validación
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Generador de datos de prueba
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Creación de los modelos
def build_model(base_model, num_classes=4):
    # Congelar las capas del modelo base
    for layer in base_model.layers:
        layer.trainable = False

    # Añadir nuevas capas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Modelos preentrenados
base_models = []
base_models.append(VGG16(include_top=False, input_shape=(150, 150, 3), weights='imagenet'))
base_models.append(InceptionV3(include_top=False, input_shape=(150, 150, 3), weights='imagenet'))
base_models.append(ResNet50(include_top=False, input_shape=(150, 150, 3), weights='imagenet'))
base_models.append(DenseNet121(include_top=False, input_shape=(150, 150, 3), weights='imagenet'))

models = [build_model(base_model) for base_model in base_models]


# Entrenamiento de los modelos por separado
history_list = []
for model in models:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=16,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size
    )
    history_list.append(history)


# Combinación de los modelos (Ensemble Learning)
def ensemble_predictions(models, generator, steps):
    generator.reset()
    sum_predictions = None

    # Hacer predicciones con cada modelo y sumarlas
    for model in models:
        predictions = model.predict(generator, steps=steps, verbose=1)
        if sum_predictions is None:
            sum_predictions = predictions
        else:
            sum_predictions += predictions

    # Calcular el promedio de las predicciones
    avg_predictions = sum_predictions / len(models)
    # Seleccionar la clase con la mayor probabilidad promedio
    final_predictions = np.argmax(avg_predictions, axis=1)
    return final_predictions


test_generator.shuffle = False
test_generator.reset()

test_steps = math.ceil(test_generator.n / test_generator.batch_size)
final_predictions = ensemble_predictions(models, test_generator, test_steps)

true_labels = test_generator.classes

ensemble_accuracy = accuracy_score(true_labels, final_predictions)
print('Accuracy of ensemble (Average Probabilities): {:.2f}%'.format(ensemble_accuracy * 100))

# Asumiendo que true_labels contiene las etiquetas verdaderas
print(classification_report(true_labels, final_predictions))

figs_dir = './figs/'

# Matriz de confusión
cm = confusion_matrix(true_labels, final_predictions)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.title('Matriz de Confusión')

# Guardar la figura en la carpeta "./figs"
conf_matrix_path = f'{figs_dir}/confusion_matrix.png'
plt.savefig(conf_matrix_path)

# Limpiar la figura actual para evitar superposiciones
plt.clf()

# Graficando resultados por modelo
for i, history in enumerate(history_list):
    plt.figure(figsize=(12, 4))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model {i+1} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model {i+1} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{figs_dir}/model_{i+1}_history.png')
    plt.close()
