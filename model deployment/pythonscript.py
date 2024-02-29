import joblib

# Load the contents of model.pkl
loaded_model = joblib.load('model.pkl')

# Check the type of the loaded object
print(type(loaded_model))
