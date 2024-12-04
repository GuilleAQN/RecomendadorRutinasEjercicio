import tensorflow as tf
import numpy as np


# def build_model(input_shape):
 #    """Defines and compiles the neural network."""

    # ToDo: Hacer una definición correcta del modelo en base a las predicciones que se van a hacer
 #    model = tf.keras.Sequential(
 #        [
 #            tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
  #           tf.keras.layers.Dense(32, activation="relu"),
 #            tf.keras.layers.Dense(1, activation="sigmoid"),
  #       ]
 #    )

 #    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
 #    return model

class ModeloRecomendacionEjercicios:
    def __init__(self):
     
        self.model = self._build_model()

    def _build_model(self):
        """
        Construye y compila el modelo de red neuronal.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(6,)),  # 6 características de entrada
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(5, activation="linear")  # 5 salidas: Ejercicio, Repeticiones, Descanso, Series, Frecuencia
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        """
        Entrena el modelo con los datos de entrada.
        """
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

    def predict(self, input_data):
    
        return self.model.predict(input_data)

    def save_model(self, filepath):
      
        self.model.save(filepath)

    def load_model(self, filepath):
        
        self.model = tf.keras.models.load_model(filepath)


def preprocess_input(nombre, edad, peso, estatura, genero, enfermedades):
    """
    Preprocesa las entradas para ser usadas en el modelo.
    """
    # Codificar género: 1 para masculino, 0 para femenino
    genero_encoded = 1 if genero.lower() == "masculino" else 0

    # Usar la cantidad de enfermedades como una característica
    enfermedades_count = len(enfermedades)

    # Crear un vector de entrada con los valores relevantes
    return np.array([edad, peso, estatura, genero_encoded, enfermedades_count, len(nombre)])

