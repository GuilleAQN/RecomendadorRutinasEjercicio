from model_training import training
from process_data import load_data
import questionary
import pandas as pd


dataset = "data/vehiculos2.csv"


def predict():
    x, y, dict_of_cast = load_data(dataset)
    model = training(x, y, debug=False)

    result = {}

    for key in dict_of_cast:
        choices = []

        for i in range(len(dict_of_cast[key]["before"])):
            choices.append(
                {
                    "name": f"{dict_of_cast[key]['before'][i]}",
                    "value": dict_of_cast[key]["after"][i],
                }
            )

        choice = questionary.select(
            f"Choose an option for {key}:", choices=choices
        ).ask()
        result[key] = choice

    choice = questionary.select(
        "ingrese el numero de puertas",
        choices=[{"name": "2", "value": 2}, {"name": "4", "value": 4}],
    ).ask()
    result["Numero de Puertas"] = choice

    choice = questionary.autocomplete(
        "Ingrese el año del vehiculo", choices=[str(i) for i in range(2000, 2024)]
    ).ask()
    result["Año"] = choice

    choice = questionary.text(
        "Ingrese el kilometraje del vehiculo",
        validate=lambda text: text.isnumeric() or "ingrese numero valido",
    ).ask()
    result["Kilometros"] = choice

    input_prec = pd.DataFrame(result, index=[0])
    input_prec = input_prec[
        [
            "Kilometros",
            "Tipo Combustible",
            "Marca",
            "Modelo",
            "Edicion",
            "Año",
            "Color",
            "Tipo de Vehiculo",
            "Estatus de Vehiculo",
            "Numero de Puertas",
        ]
    ]

    prediction = model.predict(input_prec)
    print(f"Prediction: {prediction}")


predict()
