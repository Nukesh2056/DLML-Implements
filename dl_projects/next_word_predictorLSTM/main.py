import json 
import os

curr_path = os.path.dirname(os.path.abspath(__file__))
print(curr_path)

with open(f"{curr_path}/data.json", 'r') as d:
    data = json.load(d)
    
print(data) # dictionary


