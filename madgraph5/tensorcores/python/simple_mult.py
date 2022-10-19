import numpy as np
import sys

vec = np.array([np.clongdouble(-0.0003517103775116171),
                np.clongdouble(-0.00030037661353104903),
                np.clongdouble(0.0011763748438148418),
                np.clongdouble(-0.0007079830110272223),
                np.clongdouble(0.0007625299074201143),
                np.clongdouble(-0.0006498882543260813),
                np.clongdouble(-8.014405416349263e-05),
                np.clongdouble(0.00027646320688074687),
                np.clongdouble(-0.00029955639405043364),
                np.clongdouble(0.000710710726239164),
                np.clongdouble(-0.0007888961814515527),
                np.clongdouble(0.0007616681630115399),
                np.clongdouble(-0.0005786667310520224),
                np.clongdouble(0.0007285208314475319),
                np.clongdouble(0.00011066204340233959),
                np.clongdouble(-0.0007398268927344146),
                np.clongdouble(-0.0002547514933840721),
                np.clongdouble(0.0002856081604574371),
                np.clongdouble(-0.0001041663648155121),
                np.clongdouble(0.0002841730834585424),
                np.clongdouble(0.0006070580952630303),
                np.clongdouble(-0.001209356855752769),
                np.clongdouble(7.35737561067638e-05),
                np.clongdouble(0.0003576869656674286)])

mat = np.array([
    [512, -64, -64, 8, 8, 80, -64, 8, 8, -1, -1, -10, 8, -1, 80, -10, 71, 62, -1, -10, -10, 62, 62, -28], # noqa
    [-64, 512, 8, 80, -64, 8, 8, -64, -1, -10, 8, -1, -1, -10, -10, 62, 62, -28, 8, -1, 80, -10, 71, 62], # noqa
    [-64, 8, 512, -64, 80, 8, 8, -1, 80, -10, 71, 62, -64, 8, 8, -1, -1, -10, -10, -1, 62, -28, -10, 62], # noqa
    [8, 80, -64, 512, 8, -64, -1, -10, -10, 62, 62, -28, 8, -64, -1, -10, 8, -1, -1, 8, 71, 62, 80, -10], # noqa
    [8, -64, 80, 8, 512, -64, -1, 8, 71, 62, 80, -10, -10, -1, 62, -28, -10, 62, -64, 8, 8, -1, -1, -10], # noqa
    [80, 8, 8, -64, -64, 512, -10, -1, 62, -28, -10, 62, -1, 8, 71, 62, 80, -10, 8, -64, -1, -10, 8, -1], # noqa
    [-64, 8, 8, -1, -1, -10, 512, -64, -64, 8, 8, 80, 80, -10, 8, -1, 62, 71, -10, 62, -1, -10, -28, 62], # noqa
    [8, -64, -1, -10, 8, -1, -64, 512, 8, 80, -64, 8, -10, 62, -1, -10, -28, 62, 80, -10, 8, -1, 62, 71], # noqa
    [8, -1, 80, -10, 71, 62, -64, 8, 512, -64, 80, 8, 8, -1, -64, 8, -10, -1, 62, -28, -10, -1, 62, -10], # noqa
    [-1, -10, -10, 62, 62, -28, 8, 80, -64, 512, 8, -64, -1, -10, 8, -64, -1, 8, 71, 62, -1, 8, -10, 80], # noqa
    [-1, 8, 71, 62, 80, -10, 8, -64, 80, 8, 512, -64, 62, -28, -10, -1, 62, -10, 8, -1, -64, 8, -10, -1], # noqa
    [-10, -1, 62, -28, -10, 62, 80, 8, 8, -64, -64, 512, 71, 62, -1, 8, -10, 80, -1, -10, 8, -64, -1, 8], # noqa
    [8, -1, -64, 8, -10, -1, 80, -10, 8, -1, 62, 71, 512, -64, -64, 8, 8, 80, 62, -10, -28, 62, -1, -10], # noqa
    [-1, -10, 8, -64, -1, 8, -10, 62, -1, -10, -28, 62, -64, 512, 8, 80, -64, 8, -10, 80, 62, 71, 8, -1], # noqa
    [80, -10, 8, -1, 62, 71, 8, -1, -64, 8, -10, -1, -64, 8, 512, -64, 80, 8, -28, 62, 62, -10, -10, -1], # noqa
    [-10, 62, -1, -10, -28, 62, -1, -10, 8, -64, -1, 8, 8, 80, -64, 512, 8, -64, 62, 71, -10, 80, -1, 8], # noqa
    [71, 62, -1, 8, -10, 80, 62, -28, -10, -1, 62, -10, 8, -64, 80, 8, 512, -64, -1, 8, -10, -1, -64, 8], # noqa
    [62, -28, -10, -1, 62, -10, 71, 62, -1, 8, -10, 80, 80, 8, 8, -64, -64, 512, -10, -1, -1, 8, 8, -64], # noqa
    [-1, 8, -10, -1, -64, 8, -10, 80, 62, 71, 8, -1, 62, -10, -28, 62, -1, -10, 512, -64, -64, 8, 8, 80], # noqa
    [-10, -1, -1, 8, 8, -64, 62, -10, -28, 62, -1, -10, -10, 80, 62, 71, 8, -1, -64, 512, 8, 80, -64, 8], # noqa
    [-10, 80, 62, 71, 8, -1, -1, 8, -10, -1, -64, 8, -28, 62, 62, -10, -10, -1, -64, 8, 512, -64, 80, 8], # noqa
    [62, -10, -28, 62, -1, -10, -10, -1, -1, 8, 8, -64, 62, 71, -10, 80, -1, 8, 8, 80, -64, 512, 8, -64], # noqa
    [62, 71, -10, 80, -1, 8, -28, 62, 62, -10, -10, -1, -1, 8, -10, -1, -64, 8, 8, -64, 80, 8, 512, -64], # noqa
    [-28, 62, 62, -10, -10, -1, 62, 71, -10, 80, -1, 8, -10, -1, -1, 8, 8, -64, 80, 8, 8, -64, -64, 512]  # noqa
])

# for x in range(24):
#     for y in range(24):
#         if x != y:
#             mat[x,y] = 0

for x in range(24):
    for y in range(24):
        sys.stdout.write(str(mat[x, y]) + ', ')
    print()

print(vec)
res1 = np.matmul(mat, vec)
print (np.matmul(vec, res1))