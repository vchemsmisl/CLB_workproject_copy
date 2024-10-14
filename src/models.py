import wget
import os
import zipfile
from pathlib import Path

# Dasha's project path: C:\pyproj\CLB_workproject
# Danila's project path: C:\Users\vchemsmisl\Desktop\programming\pyproj\CLB_workproject
project_path = r'C:\Users\vchemsmisl\Desktop\programming\pyproj\CLB_workproject'

url = 'http://vectors.nlpl.eu/repository/20/213.zip'
models_dir = Path(f'{project_path}\models')
if not models_dir.exists():
    models_dir.mkdir(parents=True)

if not os.listdir(f'{project_path}\models'):
    wget.download(url, out=f'{project_path}\models')

filename = Path(f'{project_path}\models\{213}.zip')
geowac_path = Path(f'{project_path}\models\geowac')

if not os.path.exists(geowac_path):
    os.makedirs(geowac_path)

with zipfile.ZipFile(filename, 'r') as zip_file:
    zip_file.extractall(geowac_path)
