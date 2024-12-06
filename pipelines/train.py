import os
from models.model_definition import ModeloRecomendacion
from pipelines.processing_data import normalize_data

def train_model(data, labels, save_dir="models/trained"):
    """
    Entrena un modelo, guarda los pesos y devuelve el modelo entrenado.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Normalizar los datos
    data = normalize_data(data)

    # Crear y entrenar el modelo
    modelo = ModeloRecomendacion()
    modelo.train(data, labels, epochs=10, batch_size=32, validation_split=0.2)

    # Guardar el modelo entrenado
    modelo.save_model(os.path.join(save_dir, "model.h5"))

    return modelo
