import os
import tensorflow as tf


class ModeloRecomendacion:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        """
        Construye y compila el modelo de red neuronal.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128, activation="relu", input_shape=(6,)
                ),  # 6 características de entrada
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(
                    5, activation="linear"
                ),  # 5 salidas: Ejercicio, Repeticiones, Descanso, Series, Frecuencia
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def load_model(self, filepath: str = "models/trained"):
        """
        Carga un modelo previamente guardado desde el archivo especificado.
        """
        self.model = tf.keras.models.load_model(filepath)

    def save_model(self, filepath: str = "models/trained"):
        """
        Guarda el modelo en el archivo especificado.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def train(self, data, labels, epochs=10, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo con los datos proporcionados.
        """
        self.model.fit(
            data, labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1
        )
        self.save_model()

