import numpy as np
import tensorflow as tf
from processing_data import process_data


class PrediccionRutinas:
    def __init__(self, model_path):
        self.model = None
        self.cargar_modelo(model_path)

    def load_custom_model(self, model_path):
        try:
            self.modelo = tf.keras.models.load_model(model_path)
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    def predict(self, model_path, input_data):
        """Load the trained model and make predictions."""
        model = self.load_custom_model(model_path)
        input_data = process_data(input_data)
        predictions = model.predict(input_data)
        return predictions.tolist()

    def interpretar_resultado(self, prediccion):
        rutinas = [
            "Cardio (Correr,Ciclismo,Natacion,Clases de Zumba)",
            "Entrenamiento de fuerza (Levantamiento de Pesas, Ejercicios con peso corporal, Entrenamientos con bandas elasticas)",
            "HIIT (Sprint, Circuitos de burpees)",
            "Flexibilidad y Movilidad (Estiramientos, Yogas, Pilates)",
            "Rutinas Funcionales (Pesas libres, Uso de herramientas de balones medicinales)",
            "Rutinas de Deportes (Atletismo, Voleibol, Futbol)",
            "Entrenamiento en circuito (Flexiones, Saltos, Abdominales)",
            "Entrenamiento de Resistencia Muscular (Series largas con peso moderado, Actividades como remo o ciclismo a larga distancia)",
        ]
        indice = np.argmax(prediccion)
        return rutinas[indice] if 0 <= indice < len(rutinas) else "Rutina desconocida"
