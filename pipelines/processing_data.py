import numpy as np

def process_data(input_data):
        # ToDo: Correct the data to process
        process_data = np.array([
            input_data["edad"],
            input_data["peso"],
            input_data["altura"]
        ])
        return process_data.reshape(1, -1)


def normalize_data(data):
    # ToDo: Do a better normalization of the data
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
