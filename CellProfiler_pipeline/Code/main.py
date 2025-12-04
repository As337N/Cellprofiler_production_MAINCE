from pathlib import Path
import functions as f
import sys

path = Path(sys.argv[1])
output = Path(sys.argv[2])
illum = bool(int(sys.argv[3]))

f.prepare_CSVs(path=path, output=output, illum=illum)
