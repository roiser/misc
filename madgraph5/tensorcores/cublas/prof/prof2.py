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

        prec = 'single'
        if self.raw[0]['precision'] == 64:
            prec = 'double'
        numcol = str(self.raw[0]['numcolors'])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3,1]))
        ax1.set_title('color matrix (%sx%s), %s precision' % (numcol, numcol, prec))
        ax1.plot(range(len(xdata)), ycub, color='tab:blue', label='cublas')
        ax1.plot(range(len(xdata)), yorg, color='tab:orange', label='original')
        ax1.set_yscale('log')
        ax1.set_ylabel('runtime (seconds)')
        ax1.legend(loc="upper left")

        ax2.set_xlabel('gridsize (#events)')
        ax2.set_ylabel('factor (N)')
        ax2.bar(range(len(xdata)), bar_factors, label=bar_labels, color=bar_colors)
        # h, l = ax2.get_legend_handles_labels()
        # l = list(map(lambda x: x.replace('cublas', 'cublas factor N better').replace('original', 'original factor N better'), l3))
        # ax2.legend(h, l, loc='upper right')
        ax2.set_xticks(range(len(xdata)), xdata, fontsize=8, rotation=66)

        fig.subplots_adjust(hspace=0)
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
