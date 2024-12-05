import tensorflow as tf


class ModeloRecomendacionEjercicios:
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

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
