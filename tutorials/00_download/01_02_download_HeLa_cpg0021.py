import subprocess
import time
import requests

path = '/fse/DC/CP_data/feature_data/20210422_6W_CP257'

Plate_Num = ['B','D','F','H','J', 'K', 'L','N']

for plate_num in Plate_Num:

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries = 20)
    sess.mount('https://', adapter)

    link = f"https://cellpainting-gallery.s3.amazonaws.com/cpg0021-periscope/broad/workspace/profiles/HeLa/single_cell/20210422_6W_CP257_single_cell_profiles_ALLBATCHES___CP257{plate_num}___ALLWELLS.csv.gz"

    try:
        subprocess.run(["wget", "-P", path, link], check=True)
    except subprocess.CalledProcessError:
        time.sleep(3)
        print("Download failed; retrying after 3 seconds.")
        subprocess.run(["wget", "-P", path, link], check=True)
