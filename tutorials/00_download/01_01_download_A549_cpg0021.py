import subprocess
import time
import random
import requests



path = '/fse/DC/CP_data/feature_data/20200805_A549_WG_Screen'

Plate_Num = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']

for plate_num in Plate_Num:
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries = 20)
    sess.mount('https://', adapter)

    link = f"https://cellpainting-gallery.s3.amazonaws.com/cpg0021-periscope/broad/workspace/profiles/A549/single_cell/20200805_A549_WG_Screen_single_cell_profiles_ALLBATCHES___CP186{plate_num}___ALLWELLS.csv.gz"

    try:
        subprocess.run(["wget", "-P", path, link], check=True)
    except subprocess.CalledProcessError:
        time.sleep(3)
        print("Download failed; retrying after 3 seconds.")
        subprocess.run(["wget", "-P", path, link], check=True)
