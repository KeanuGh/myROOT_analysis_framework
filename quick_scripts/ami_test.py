import os

import pyAMI.atlas.api as ami
import pyAMI.client
from dotenv import load_dotenv

load_dotenv()
CERT_PATH = os.getenv('CERT_FILE')
KEY_PATH = os.getenv('KEY_FILE')

client = pyAMI.client.Client('atlas', cert_file=CERT_PATH, key_file=KEY_PATH, ignore_proxy=True, verbose=True)

# extract data
# df = uproot.concatenate('../data/wminmunu_MC.root:sumWeights')

# result = client.execute('list datasets --dataset-number 301170 -f cross_section,nfiles,physics_short,events,total_size', format='dict_object')
kwargs = {'dataset-number': 301170}
result = ami.get_dataset_info(client, **kwargs)
print(result)
