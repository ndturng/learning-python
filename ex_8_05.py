f_name = input('Enter a file name: ')

try:
    f_open = open(f_name)
except:
    print('Can not open', f_name)
    exit()

count = 0

for line in f_open:
    if line.startswith('From:'):
        continue
    elif line.startswith('From'):
        count = count + 1    
        line = line.rstrip()
        words = line.split()
    try:
        print(words[1])
    except: continue

print('There were', count,'lines in the file with From as the first word')