from prefect import flow
from data.preprocessing import prepare_data
from models.training import train_model

@flow(name="Training Pipeline")
def training_pipeline():
    # Data preparation
    train_data, val_data = prepare_data()
    
    # Model training
    model = train_model(
        train_data,
        learning_rate=0.0001,
        epochs=10
    )
    
    # Model evaluation
    results = model.evaluate(val_data)
    
    # Model export
    model.export("models/production/")

if __name__ == "__main__":
    training_pipeline()