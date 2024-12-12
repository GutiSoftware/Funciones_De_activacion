import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time
import os
import json

# Crear carpeta "Resultados" en la ubicación del script
script_dir = os.path.dirname(os.path.abspath(__file__))
resultados_dir = os.path.join(script_dir, "Resultados")
os.makedirs(resultados_dir, exist_ok=True)

#ATENCIÓN!! ATENCIÓN!! ATENCIÓN!!
#Dejar solo una funcion custom actiVa, poniendo en comentario todas las demás

"""def custom(x):
    #Función de activación personalizada: WaveSoft.
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "WaveSoft"
    formula = "f(x) = x * sin(x)"
    return x * tf.sin(x), function_name, formula"""

"""def custom(x):
    #Función de activación personalizada: SymSoftReLU.
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "SymSoftReLU"
    formula = "f(x) = log(1 + e^(|x|)) * sign(x)"
    return tf.math.log(1 + tf.exp(tf.abs(x))) * tf.sign(x), function_name, formula"""

"""def custom(x):
    #Función de activación personalizada: SinShift
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "SinShift"
    formula = "f(x) = sin(x) + 0.5 * x"
    return tf.sin(x) + 0.5 * x, function_name, formula"""

"""def custom(x):
    #Función de activación personalizada: RootSquareSoft
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "RootSquareSoft"
    formula = "f(x) = sqrt(|x|) * sign(x) + x^2"
    return tf.sqrt(tf.abs(x)) * tf.sign(x) + x ** 2, function_name, formula"""

"""def custom(x):
    #Función de activación personalizada: LogCosh.
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "LogCosh"
    formula = "f(x) = log(cosh(x))"
    return tf.math.log(tf.cosh(x)), function_name, formula"""

def custom(x):
    """Función de activación personalizada basada en la fórmula f(x) = m1 * x + (m2 - m1) * x * sigmoid(k * x)."""
    x = tf.cast(x, tf.float32)  # Asegurar que x sea de tipo flotante
    function_name = "CustomActivation"  # Nombre de la función personalizada
    formula = "f(x) = m1 * x + (m2 - m1) * x * sigmoid(k * x)"  # Fórmula de la función

    # Definir parámetros m1, m2 y k
    # m1 = 0.01  # Puedes ajustar este valor según sea necesario
    # m2 = 1  # Puedes ajustar este valor según sea necesario
    # k = 2.0   # Puedes ajustar este valor según sea necesario

    # Calcular el término sigmoidal
    sigmoid_term = 1 / (1 + tf.exp(-2.0 * x))  # Término sigmoid(k * x)

    # Calcular la fórmula completa
    result = 0.01 * x + 0.99 * x * sigmoid_term

    return result, function_name, formula




# Lista de funciones de activación a probar
activation_functions = {
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
}

# Cargar el dataset MNIST
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    # Normalizar los datos
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # Redimensionar para compatibilidad con capas convolucionales
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    return (train_images, train_labels), (test_images, test_labels)

# Crear el modelo con la función de activación especificada
def create_model(activation_function):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=activation_function, input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation_function),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation_function),
        layers.Flatten(),
        layers.Dense(64, activation=activation_function),
        layers.Dense(10, activation='softmax')  # Output layer para clasificación
    ])
    return model

# Entrenar el modelo
def train_model(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(train_images, train_labels, epochs=5, 
                        validation_data=(test_images, test_labels), verbose=2)
    training_time = time.time() - start_time
    return history, training_time

# Guardar datos en un archivo JSON
def save_results(data, filename):
    filepath = os.path.join(resultados_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Cargar datos desde un archivo JSON
def load_results(filename):
    filepath = os.path.join(resultados_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# Visualizar los resultados
def plot_results(history, test_acc, training_time, filename):
    plt.figure(figsize=(12, 5))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Entrenamiento')
    plt.plot(history['val_accuracy'], label='Validación')
    plt.title(f'Precisión ({filename})\nTest Accuracy: {test_acc:.4f}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Entrenamiento')
    plt.plot(history['val_loss'], label='Validación')
    plt.title(f'Pérdida ({filename})\nTiempo de entrenamiento: {training_time:.2f} s')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Guardar las gráficas en la carpeta "Resultados"
    graph_path = os.path.join(resultados_dir, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

# Main: Entrenar y evaluar todas las funciones de activación
if __name__ == "__main__":
    # Cargar los datos
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Agregar la función personalizada a la lista
    custom_output, custom_name, custom_formula = custom(0.0)  # Llamada con un valor flotante
    activation_functions[custom_name] = lambda x: custom(x)[0]

    for function_name, activation_function in activation_functions.items():
        # Nombre del archivo para guardar los resultados
        data_filename = f"results_{function_name}.json"
        graph_filename = f"results_{function_name}"

        # Verificar si los resultados ya existen
        existing_results = load_results(data_filename)
        if existing_results:
            print(f"Resultados existentes encontrados para '{function_name}'. No se recalcularán.")
            print(f"Precisión en el conjunto de prueba: {existing_results['test_accuracy']}")
            print(f"Tiempo de entrenamiento: {existing_results['training_time']:.2f} segundos")
        else:
            # Entrenar y evaluar el modelo
            print(f"\nEntrenando modelo para la función '{function_name}'...")
            model = create_model(activation_function)
            history, training_time = train_model(model, train_images, train_labels, test_images, test_labels)

            # Evaluar el modelo en el conjunto de prueba
            test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
            print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")
            print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

            # Guardar los datos en un archivo JSON
            results_data = {
                "function": function_name,
                "formula": custom_formula if function_name == custom_name else "Standard Activation",
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "training_time": training_time,
                "history": {
                    "accuracy": history.history['accuracy'],
                    "val_accuracy": history.history['val_accuracy'],
                    "loss": history.history['loss'],
                    "val_loss": history.history['val_loss']
                }
            }
            save_results(results_data, data_filename)

            # Generar y guardar las gráficas
            plot_results(results_data['history'], test_acc, training_time, graph_filename)
