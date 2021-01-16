import csv

def load_csv2(file_path, l):
    """
    type(file_path) = str()
    type(l) = list()
    csv file with 2 columns separated with comma
    Loading function. Appending data to l as [row[0],row[1]]
    """
    with open(file_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                row[0] = row[0].replace("<br /><br />", " ")
                l.append([row[0], int(row[1])])
                line_count += 1
            percent = (line_count)/50000*100
            if percent%5 == 0 and line_count != 0:
                print(f'Parsed {percent:.1f}% ({line_count}) rows of {file_path}')
        print('Done\n')
    return

def pn_compare(l):
    pos, neg = 0, 0
    for rev in l:
        if rev[1] == 0:
            neg += 1
        else:
            pos += 1
    print(f"Ilosć recenzji:\nPozytywnych: {pos}\nNegatywnych: {neg}")
    if pos == neg:
        print("Dane są zbalansowane.")

a = []
load_csv2("movies_data.csv", a)
pn_compare(a)

###############
# inna opcja ładowania danych ale ma problem ze znakami \ :
#
#reviews = []
#f = open("movies_data.csv", "r", encoding='utf-8')
#f.readline()
#for line in f:
#    line = line.replace("<br /><br />", " ")
#    line = line.replace("\n", "")
#    line = line.replace("\\'", "'")            # ale to nie działa :(
#    reviews.append([line[:-2],line[-1]])
#f.close()