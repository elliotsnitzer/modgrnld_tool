#!/usr/bin/env python

import os, sys, time

dir_path = '/data/groups/ghub/tools/modgrnld'
dir_options = ['MODGRNLD', 'MERRA-2']

def bytesto(bytes, to, bsize=1024):
    """convert bytes to megabytes, etc.
       sample code:
           print('mb= ' + str(bytesto(314575262000000, 'm')))
       sample output: 
           mb= 300002347.946
    """

    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return(r)

#calculate size of MODGRNLD and MERRA-2 directories within data directory
modgrnld_size = 0
merra_size = 0
for opt in dir_options:
    size = 0
    dp = os.path.join(dir_path, opt)
    for path, dir, files in os.walk(dp):
        for f in files:
            filename = os.path.join(path, f)
            size+=os.path.getsize(filename)
    if(opt==dir_options[0]):
        modgrnld_size = size
    else:
        merra_size = size
       
    
total_size = modgrnld_size+merra_size
            
        
size_limit = 10000000000000 #temporary value, limit not decided yet. Limit in bytes
now = time.time()
if(modgrnld_size>size_limit):
    dp = os.path.join(dir_path, dir_options[0])
    for path, dir ,files in os.walk(dp):
        for f in files:
            filename = os.path.join(path,f)
            if(os.stat(filename).st_mtime < now-(7*86400)):
                #file is more than a week old
                #os.remove(filename)
    
if(merra_size>size_limit):
    dp = os.path.join(dir_path, dir_options[1])
    for path, dir ,files in os.walk(dp):
        for f in files:
            filename = os.path.join(path,f)
            if(os.stat(filename).st_mtime < now-(7*86400)):
                #file is more than a week old
                #os.remove(filename)
        

size_label = ''
convert_size = total_size
if(total_size>1000 and total_size<1000000):
    convert_size = bytesto(total_size, 'k')
    size_label = 'kilo'
elif(total_size>1000000 and total_size<1000000000):
    convert_size = bytesto(total_size, 'm')
    size_label = 'mega'
else:
    convert_size = bytesto(total_size, 'g')
    size_label = 'giga'
    

print('MODGRNLD Directory Size: {} bytes'.format(modgrnld_size))
print('MERRA Directory Size: {} bytes'.format(merra_size))
print('Total Directory Size: {:.2f} {}bytes'.format(convert_size, size_label))