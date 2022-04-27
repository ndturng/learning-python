list_num = []

while True:
    x = input('Enter a number: ')
    if x == 'done':
        break
    try:
        x = float(x)
    except:
        print('not a number')
        continue
    list_num.append(x)

print('Maximum:', max(list_num))
print('Minimum:', min(list_num))