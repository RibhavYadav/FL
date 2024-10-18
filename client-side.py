"""
'Client' side implementation of the devices connected in Federated Learning
Client class with functions for it's working:
    - receive_global: receives the parameters from the global model along with other parameters
        - weights: array like
        - batch_size: minimum number of rows in a batch
    - calculate_local: initialise model and train it for a fixed batch size
        - initialise_model: start a local Linear Regression model
        - split_data: split the data into the given number of batch size
        - train_model: train the model on the local data for 1 batch
    - send_local: sends the updated model parameters back to the global model
"""
