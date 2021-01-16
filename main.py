from part_A import load_csv2 as lcsv
from part_A import pn_compare as pnc

a = []
lcsv("movies_data.csv", a)
pnc(a)


