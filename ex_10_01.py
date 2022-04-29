f_name = input('Enter a file name: ')
# if len(f_name) == 0:
#     f_name = 'mbox-short.txt'
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


 
inverse = [(value, key) for key, value in lst.items()]
print(max(inverse)[1], max(inverse)[0])

