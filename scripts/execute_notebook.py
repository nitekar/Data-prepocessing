import nbformat
from nbclient import NotebookClient
import sys

NB_PATH = 'notebooks/product_recommendation.ipynb'
TIMEOUT = 600

print(f'Executing notebook: {NB_PATH} (timeout={TIMEOUT}s)')
nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=TIMEOUT, kernel_name='python3')
try:
    client.execute()
    nbformat.write(nb, NB_PATH)
    print('Notebook executed and saved in-place.')
except Exception as e:
    print('Notebook execution failed:', e)
    sys.exit(1)
