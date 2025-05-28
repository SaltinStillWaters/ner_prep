import pickle

with open('processed_data/lime_stg_2.pkl', "rb") as f:
    data = pickle.load(f)

# Check the type
print(type(data))

# View keys if it's a dict
if isinstance(data, dict):
    print(data.keys())

# Preview the content
print(data)