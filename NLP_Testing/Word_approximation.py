from fuzzywuzzy import fuzz as fz
from fuzzywuzzy import process as pr
import csv

def fuzzy_matching(text):

    #Declarations
    Fname_ds = []
    Lname_ds = []
    body_part = ['abdomen', 'barium', 'bone', 'chest', 'dental', 'extremity', 'hand', 'joint', 'neck', 'pelvis', 'sinus', 'skull', 'spine', 'thoracic']

    #Opening CSV files
    with open('/home/oxidane/Desktop/XRAY/X_Ray_Data_Detector-main/Alpha_Testing/Indian_Names.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        
        for e in reader:
            Fname_ds.append(e[1])

    with open('/home/oxidane/Desktop/XRAY/X_Ray_Data_Detector-main/Alpha_Testing/indian_last_name.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)

        for e in reader:
            Lname_ds.append(e[0])


    if len(text) < 3 or pr.extractOne(text, body_part)[1] > 75: return False

    #Checking for first name
    for given in Fname_ds:
        if fz.ratio(text, given) > 75:
            print(text, given, 1)
            return True


    #Checking for last name
    for given in Lname_ds:
        if fz.ratio(text, given) > 75:
            print(text, given, 2)
            return True
        

    return False