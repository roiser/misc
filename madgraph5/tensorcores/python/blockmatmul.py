import numpy as np

# https://de.wikipedia.org/wiki/Matrizenmultiplikation#Blockmatrizen
# https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_multiplication


class BlockMatrixMult():

    def __init__(self, msize, mstride):
        self.msize = msize
        self.mstride = mstride
        self.ma = np.array()
        self.mb = np.array()
        self.mr = np.array()

    def fillmatrix(self):
        pass

    def fullmultiplication(self):
        pass

    def blockmultiplication(self):
        pass

    def check(self):
        pass

    def run(self):
        self.fillmatrix()
        self.fullmultiplication()
        self.blockmultiplication()
        self.check()


if __name__ == '__main__':
    bmm = BlockMatrixMult(8, 8)
    bmm.run()
