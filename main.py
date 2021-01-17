from modules.part_A import load_csv2 as lcsv
from modules.part_A import pn_compare as pnc

a = []
a = lcsv("movies_data.csv", a, lim = 1020)
pnc(a)


