import datasets as d
import random as r
import csv

instruction = "You are a latex autocomplete model. You will be given a setence from a proof and you need to finish the sentence. Give back the setence in latex markup."
path = './dataset/latex.txt'

def main():
  with open(path, 'r') as file:
      data = file.read()


  data = data.split('.')
  data = [x.strip('\n\\n').replace('\n', ' ').replace('  ', ' ') for x in data]

  dataset = []
  cutoff = 7
  lb = 0.42
  up = 0.66
  for i in range(len(data)):
    length = len(data[i])
    if length < cutoff:
      continue
    index = r.randint(int(lb * length), int(up * length))
    dataset.append({
      'instruction': instruction,
      'input': data[i][:index],
      'output': data[i][index:]
    })


  with open('./dataset/latex.csv', mode='w') as file:
    writer = csv.DictWriter(file, fieldnames=['instruction', 'input', 'output'])
    writer.writeheader()
    writer.writerows(dataset)

def get_dataset():
  main()
  dataset = d.load_dataset('csv', data_files='./dataset/latex.csv')
  return dataset

if __name__ == '__main__':
  main()