t = 1 
b = 1

while t <= 2048:
    b = 1
    while b <= 2048:
        print("%d %d" % (t, b))
        b *= 2
    t *= 2
