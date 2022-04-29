f_name = input('Enter a file name: ')

# if len(f_name) == 0:
#     f_name = 'mbox-short.txt'

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()


di = dict()

for line in f_open:
    
    if line.startswith('From:'):
        continue
    elif line.startswith('From'):   
        line = line.rstrip()
        words = line.split()
                                
        try:
            a = words[5].split(':')
            di[a[0]] = di.get(a[0],0) + 1
        except: continue


lst = list(di.items())
lst.sort()

for i in lst:
    print(i[0], i[1])
