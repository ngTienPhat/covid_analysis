import os
import default_config
import data
import pandas as pd
#from default_config import get_default_config
from data.eda import distribution_active_cases

import sys
print('\n'.join(sys.path))

def main():
    # os.chdir(default_config.CWD)
    texas_df = pd.read_csv(os.path.join(default_config.DATA_ROOT, 'Texas.csv'))
    distribution_active_cases(texas_df)


if __name__ == "__main__":
    #cfg = get_default_config()
    main()