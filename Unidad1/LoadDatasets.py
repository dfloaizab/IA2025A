import csv

def loadAccidentDataset(filePath):

    data = []
    filePath = "accident.csv"

    with open(filePath,mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data