import pandas as ps
from sklearn.preprocessing import LabelEncoder


def load_data(csv_path):

    df = ps.read_csv(csv_path)

    df = df[
        (df["Kilometros"] >= 0)
        | ((df["Kilometros"] <= 1000) & (df["Estatus de Vehiculo"] == "Nuevo"))
    ]

    df["Tipo Combustible"] = df["Tipo Combustible"].apply(
        lambda x: "GLP" if x == "Gas" else x
    )

    df = df[df["Numero de Puertas"].isin([2, 4])]

    label_encoder = LabelEncoder()

    list_of_columns = [
        "Tipo Combustible",
        "Marca",
        "Modelo",
        "Edicion",
        "Color",
        "Tipo de Vehiculo",
        "Estatus de Vehiculo",
    ]

    dict_of_cast = {}
    for i in list_of_columns:
        dict_of_cast[i] = {}
        dict_of_cast[i]["before"] = df[i].unique()
        df[i] = label_encoder.fit_transform(df[i])
        dict_of_cast[i]["after"] = df[i].unique()

    X = df.drop("Precio", axis=1)
    y = df["Precio"]

    return X, y, dict_of_cast
