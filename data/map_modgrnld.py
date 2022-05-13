#!/usr/bin/env python

import os, sys

sys.path.append('data')
import nsidc_download_tools

# Query all data files in MODGRNLD Data Set
filename_filter = 'MODGRNLD.*.monthly.*'
MODGRNLD_url_list = nsidc_download_tools.cmr_search('MODGRNLD', filename_filter = filename_filter)
    
info_file = open("/data/groups/ghub/tools/modgrnld/file_info.txt",'w')

for url in MODGRNLD_url_list:
    filename = url.split('/')[-1]
    info_file.write(filename+'\n')
info_file.close()