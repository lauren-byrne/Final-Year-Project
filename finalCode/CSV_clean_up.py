import pandas as pd
import numpy as np

#read in dataset from csv file and store in dataframe 'df'
df = pd.read_csv("test.csv", header = None, na_values=' ?')

f = open("column_names.txt", "w+")

for i in range(68):
    f.write(str(i) + '_x\n')
    f.write(str(i) + '_y\n')

f.write('Gaze\n')
f.write('Blink')

f.close()

text_file = open("column_names.txt", "r")
#split text file into a list called new_names
new_names = text_file.read().splitlines()
text_file.close()

df.columns = new_names

df.to_csv('test2.csv', sep = ',')

