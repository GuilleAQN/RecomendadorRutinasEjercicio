from sklearn.metrics import accuracy_score, f1_score

class ModelEvaluator:
    def __init__(self, model, test_data, test_labels):
        """
        Inicializa el evaluador de modelos.

        Parameters:
        - model_path (str): Ruta del modelo Keras guardado.
        - test_data (np.ndarray): Datos de prueba para evaluar el modelo.
        - test_labels (np.ndarray): Etiquetas verdaderas correspondientes a los datos de prueba.
        """
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels

    def __predict(self):
        """Realiza predicciones sobre los datos de prueba.

        Returns:
        - np.ndarray: Etiquetas binarias predichas (0 o 1) basadas en un umbral de 0.5.
        """
        predictions = self.model.predict(self.test_data)
        # Convertir probabilidades a etiquetas binarias
        return (predictions > 0.5).astype("int32")

    def evaluate_accuracy(self):
        """Evalúa la precisión del modelo.

        La precisión se calcula utilizando la función accuracy_score de sklearn. 
        Esta métrica mide la proporción de predicciones correctas (verdaderos positivos 
        y verdaderos negativos) sobre el total de predicciones realizadas. Es útil 
        para tener una idea general del rendimiento del modelo, especialmente cuando 
        las clases están balanceadas.

        Returns:
        - dict: Un diccionario que contiene la precisión del modelo.
                Ejemplo: {"accuracy": 0.95}
        """
        predictions = self.__predict()
        accuracy = accuracy_score(self.test_labels, predictions)
        return {"accuracy": accuracy}

    def evaluate_f1_score(self):
        """Evalúa el F1-score del modelo.

        El F1-score se calcula utilizando la función f1_score de sklearn. 
        Esta métrica es la media armónica entre la precisión y el recall (sensibilidad). 
        Es particularmente útil en situaciones donde hay un desbalance en las clases, 
        ya que proporciona una única métrica que considera tanto los falsos positivos 
        como los falsos negativos. Un F1-score alto indica un buen equilibrio entre 
        precisión y recall.

        Returns:
        - dict: Un diccionario que contiene el F1-score del modelo.
                Ejemplo: {"f1_score": 0.85}
        """
        predictions = self.__predict()
        f1 = f1_score(self.test_labels, predictions)
        return {"f1_score": f1}
