"""
'Server' side implementation for Federated Learning
Server class with functions for it's working:
    - initialise_global: creates the parameters for the global Linear Regression model
    - connect_clients: connects to the clients that have the data
        - This is done by create client-side objects
        - send_data: sends the parameters of the model to the clients
    - aggregate_parameters: aggregates the parameters sent by the clients
        - Receives the parameters from the clients
        - aggregates the parameters through Federated Averaging
    - update_parameters: updates the parameters for the global model
"""
