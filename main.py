from pipelines.train import train_model
from pipelines.model_prediction import predict
from pipelines.evaluate_model import evaluate_model
from utils.data_loader import load_data, split_data

if __name__ == "__main__":
    # Load and split data
    data = load_data("data/processed/train.csv")
    X, y = split_data(data, target_column="label")
    
    # Train and save the model
    train_model(X, y)

    # Load test data
    test_data = load_data("data/processed/test.csv")
    X_test, y_test = split_data(test_data, target_column="label")
    
    # Evaluate the model
    metrics = evaluate_model("models/trained/model.h5", X_test, y_test)
    print(metrics)

    input_data = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    
    # Load model and predict
    predictions = predict("models/trained/model.h5", input_data)
    print("Predictions:", predictions)
