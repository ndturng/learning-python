import string

fname = input('Enter the file name: ')

try:
    fhand = open(fname)
except:
    print('File cannot be opened:', fname)
    exit()

counts = dict()
for line in fhand:
    line = line.rstrip()    
    line = line.translate(line.maketrans('', '', string.punctuation))
    line = line.lower()    
    words = line.split()
    
    for word in words:
        if word[0] in '0123456789': continue
        counts[word] = counts.get(word,0) +1
        

lst = list((val, key) for key, val in counts.items())         
lst.sort(reverse=True)

for i in lst:
    print(i[1], i[0])
