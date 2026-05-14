from pathlib import Path
import functions as f
import sys

path = Path(sys.argv[1])
output = Path(sys.argv[2])
name_csv = str(sys.argv[3])
illum = bool(int(sys.argv[4]))
masks = bool(int(sys.argv[5]))
plates2process = 

f.prepare_CSVs(path=path, output=output, name_csv=name_csv, illum=illum, masks=masks)