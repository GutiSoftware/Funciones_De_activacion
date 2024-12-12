import os
import json
import pandas as pd

# Ruta de la carpeta donde están los archivos JSON
resultados_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resultados")

# Modelos estándar a incluir en la comparación
standard_models = ["relu", "sigmoid", "tanh"]

def load_json_files():
    """Carga todos los archivos JSON de la carpeta Resultados."""
    data = {}
    for filename in os.listdir(resultados_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(resultados_dir, filename)
            with open(filepath, "r") as f:
                model_data = json.load(f)
                model_name = model_data["function"]
                data[model_name] = model_data
    return data

def compare_models(data):
    """Crea una tabla comparativa para cada modelo personalizado."""
    # Filtrar los modelos personalizados
    custom_models = [model for model in data.keys() if model not in standard_models]

    for custom_model in custom_models:
        print(f"\nGenerando tabla comparativa para el modelo '{custom_model}'...\n")
        
        # Crear una lista para almacenar los datos de comparación
        comparison_data = []

        # Agregar los datos del modelo personalizado
        custom_data = data[custom_model]
        comparison_data.append({
            "Model": custom_model,
            "Formula": custom_data.get("formula", "N/A"),
            "Test Accuracy": custom_data["test_accuracy"],
            "Test Loss": custom_data["test_loss"],
            "Training Time (s)": custom_data["training_time"]
        })

        # Agregar los datos de los modelos estándar
        for standard_model in standard_models:
            if standard_model in data:
                standard_data = data[standard_model]
                comparison_data.append({
                    "Model": standard_model,
                    "Formula": "Standard Activation",
                    "Test Accuracy": standard_data["test_accuracy"],
                    "Test Loss": standard_data["test_loss"],
                    "Training Time (s)": standard_data["training_time"]
                })

        # Convertir los datos a un DataFrame de pandas
        df = pd.DataFrame(comparison_data)

        # Mostrar la tabla en pantalla
        print(df)

        # Guardar la tabla como un archivo CSV
        csv_filename = os.path.join(resultados_dir, f"comparison_{custom_model}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Tabla guardada en: {csv_filename}\n")

if __name__ == "__main__":
    # Cargar los datos desde los archivos JSON
    data = load_json_files()

    # Crear las tablas comparativas
    compare_models(data)
