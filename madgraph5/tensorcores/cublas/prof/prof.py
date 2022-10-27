import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import json
import os
import sys

class prof():

    def __init__(self, prec, ncol):
        self.prec = prec
        self.ncol = ncol
        self.raw = []
        self.data = {}


    def read_data(self):
        dir = '%s_%s' % (prec, ncol)
        for f in os.listdir(dir):
            fh = open(dir + '/' + f)
            self.raw.append(json.load(fh))
            fh.close()

    def prepare(self):
        bar_factors = []
        bar_labels = []
        bar_colors = []
        for r in self.raw:
            nev = r['numevents']
            if nev not in self.data:
                self.data[nev] = {}
                self.data[nev]['cub'] = []
                self.data[nev]['org'] = []
            cavg = r['cublas']['avg']
            davg = r['device']['avg']
            if davg > 1e-5 and cavg > 1e-5:
                self.data[nev]['cub'].append(cavg)
                self.data[nev]['org'].append(davg)

        xdata = list(self.data.keys())
        xdata.sort()

        ycub = []
        yorg = []
        for x in xdata:
            cubl = self.data[x]['cub']
            orgl = self.data[x]['org']
            cuavg = sum(cubl)/len(cubl)
            oravg = sum(orgl)/len(orgl)
            print('%s %.2f' % (str(x).rjust(8), oravg/cuavg))
            if oravg > cuavg:
                bar_factors.append(oravg/cuavg)
                bar_colors.append('tab:blue')
                if 'cublas' not in bar_labels:
                    bar_labels.append('cublas')
                else:
                    bar_labels.append('_cublas')
            else:
                bar_factors.append(cuavg/oravg)
                bar_colors.append('tab:orange')
                if 'original' not in bar_labels:
                    bar_labels.append('original')
                else:
                    bar_labels.append('_original')
            # print(cubl)
            # print(orgl)
            # print()
            ycub.append(cuavg)
            yorg.append(oravg)

        fig = plt.figure()
        #fig, (ax,ax2) = plt.subplots(2)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(len(xdata)), ycub, color='tab:blue', label='cublas')
        ax.plot(range(len(xdata)), yorg, color='tab:orange', label='original')

        plt.yscale('log')
        plt.xticks(range(len(xdata)), xdata, rotation=45)
        plt.legend(loc="upper left")
        plt.xlabel('gridsize (#events)')
        plt.ylabel('runtime (seconds)')

        prec = 'single'
        if self.raw[0]['precision'] == 64:
            prec = 'double'
        numcol = str(self.raw[0]['numcolors'])

        ax.set_title('color matrix (%sx%s), %s precision' % (numcol, numcol, prec))

        plt.show()

        fig2, ax2 = plt.subplots()

        plt.xticks(range(len(xdata)), xdata, rotation=45)

        ax2.bar(range(len(xdata)), bar_factors, label=bar_labels, color=bar_colors)

        ax2.set_ylabel('factor (N)')
        #ax2.set_xlabel('gridsize (#events)')
        #ax2.set_title('Factors faster')
        #ax2.legend(title='')

        plt.show()


    def run(self):
        self.read_data()
        self.prepare()

def usage():
    print('%s precision colors' % sys.argv[0])
    sys.exit(1)

if __name__ == '__main__':
    precs = ['32', '64']
    ncols = ['24', '120']
    if len(sys.argv) != 3: usage()
    prec = str(sys.argv[1])
    ncol = str(sys.argv[2])
    if prec not in precs or ncol not in ncols: usage()

    prof(prec, ncol).run()
