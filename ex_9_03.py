f_name = input('Enter a file name: ')

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()

lst = dict()

for line in f_open:
    
    if line.startswith('From:'):
        continue
    elif line.startswith('From'):   
        line = line.rstrip()
        words = line.split()
        try:
            lst[words[1]] = lst.get(words[1],0) + 1
        except: continue

print(lst)
