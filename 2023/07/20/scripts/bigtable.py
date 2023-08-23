t= [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  32, 32, 32, 32, 32, 32,
      32, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32,

      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

def swap_shift(x):
  #     r = (long int)_bswap64((long long int)r);
  # uint64_t rs = ((uint64_t)r >> (3 * 8));
  z = (x & 0xff) << (7*8)
  z |= ((x>>8) & 0xff) << (6*8)
  z |= ((x>>(2*8)) & 0xff) << (5*8)
  z |= ((x>>(3*8)) & 0xff) << (4*8)
  z |= ((x>>(4*8)) & 0xff) << (3*8)
  z |= ((x>>(5*8)) & 0xff) << (2*8)
  z |= ((x>>(6*8)) & 0xff) << (1*8)
  z |= ((x>>(7*8)) & 0xff) 
  return z >> (3*8)

for i in range(8):
    table = [0 for z in range(256)]
    for j in range(256):
        x = t[j]
        if(x == 32):
            table[j] = (1 << i)<<56
        else:
            table[j] = swap_shift(x << (5 * (7 - i)))
    print("uint64_t table"+str(i)+"[] = "+str(table))

print("---")
print()
# zeroing table
z = [0,]
running = 0
for i in range(8):
    running |= swap_shift(0b11111 << (5 * (7 - i)))
    z.append(running)
print(z)

w = []
for i in range(256):
    if(t[i] != 32):
        w.append((t[i], chr(i)))
w.sort()
for (a,b) in w:
    if(b.upper() == b):
       print("<tr><td>'"+str(b)+"'</td><td>",a,"</td></tr>")
