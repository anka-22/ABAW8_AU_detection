import pickle

file1 = "/home/ckp/checkpoint1.pkl"
file2 = "/home/ckp/checkpoint2.pkl"

with open(file1, "rb") as input_file:
    e = pickle.load(input_file)

with open(file2, "rb") as input_file:
    f = pickle.load(input_file)

print(0)