import os, sys, time

dir_path = '/data/groups/ghub/tools/reference'

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

size = 0
for path, dir, files in os.walk(dir_path):
    for f in files:
        filename = os.path.join(path, f)
        print('{} Size: {}'.format(filename,os.path.getsize(filename)))
        size+=os.path.getsize(filename)
        
size_label = ''
convert_size = size
if(size>1000 and size<1000000):
    convert_size = bytesto(size, 'k')
    size_label = 'kilo'
elif(size>1000000 and size<1000000000):
    convert_size = bytesto(size, 'm')
    size_label = 'mega'
else:
    convert_size = bytesto(size, 'g')
    size_label = 'giga'
    
print('Total Directory Size: {:.2f} {}bytes'.format(convert_size, size_label))