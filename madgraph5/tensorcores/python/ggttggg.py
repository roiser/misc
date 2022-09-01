import numpy as np

ncolor = 24
stride = 8

cf = np.array([
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

denom = np.array([54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
                  54, 54, 54, 54, 54, 54, 54, 54, 54])

jamp2 = np.array([
    [np.clongdouble(-0.0003517103775116171 + 1j * -0.0006505571261836556),
     np.clongdouble(-0.00030037661353104903 + 1j * 0.0008903692101049247),
     np.clongdouble(0.0011763748438148418 + 1j * -0.00023698458114937643),
     np.clongdouble(-0.0007079830110272223 + 1j * 0.001347573459134033),
     np.clongdouble(0.0007625299074201143 + 1j * -0.0005652378475512697),
     np.clongdouble(-0.0006498882543260813 + 1j * -0.0006568838133079037),
     np.clongdouble(-8.014405416349263e-05 + 1j * -0.0006388243166556857),
     np.clongdouble(0.00027646320688074687 + 1j * 0.0003959223887304082),
     np.clongdouble(-0.00029955639405043364 + 1j * 0.0005735525642930147),
     np.clongdouble(0.000710710726239164 + 1j * 0.0006791100883304888),
     np.clongdouble(-0.0007888961814515527 + 1j * 0.0002393608137330206),
     np.clongdouble(0.0007616681630115399 + 1j * -0.0014696914067261553),
     np.clongdouble(-0.0005786667310520224 + 1j * -6.2187824781173825e-06),
     np.clongdouble(0.0007285208314475319 + 1j * -0.0002147288301571186),
     np.clongdouble(0.00011066204340233959 + 1j * -0.00034114329824648526),
     np.clongdouble(-0.0007398268927344146 + 1j * 0.0005674298886572326),
     np.clongdouble(-0.0002547514933840721 + 1j * -0.00037551988602026393),
     np.clongdouble(0.0002856081604574371 + 1j * -0.0008770490810408817),
     np.clongdouble(-0.0001041663648155121 + 1j * 0.00033554861219346283),
     np.clongdouble(0.0002841730834585424 + 1j * -0.0005220299775191181),
     np.clongdouble(0.0006070580952630303 + 1j * -1.2765828252936482e-06),
     np.clongdouble(-0.001209356855752769 + 1j * 0.0002499778435442038),
     np.clongdouble(7.35737561067638e-05 + 1j * 0.0005975180903271501),
     np.clongdouble(0.0003576869656674286 + 1j * 0.0006315702150005543)],
    [np.clongdouble(4.001096657891247e-05 + 1j * 1.4791805309750028e-05),
     np.clongdouble(-5.923938906400064e-05 + 1j * 3.801934765541291e-06),
     np.clongdouble(0.0001427374506265507 + 1j * -1.8043457422002595e-05),
     np.clongdouble(-0.0001591737775768255 + 1j * -0.00036165089219690884),
     np.clongdouble(4.225326999520912e-05 + 1j * -8.602693533068292e-06),
     np.clongdouble(-4.6584484745537194e-05 + 1j * 6.701854604158101e-05),
     np.clongdouble(0.00012226521402585174 + 1j * -9.482879230678914e-05),
     np.clongdouble(-6.565734283734101e-05 + 1j * 1.4064660080652031e-05),
     np.clongdouble(-7.132489650805094e-05 + 1j * 9.864520009223758e-05),
     np.clongdouble(-8.310498364342943e-05 + 1j * 1.5195094466812519e-05),
     np.clongdouble(-5.419267360769465e-05 + 1j * 0.00015195491448562545),
     np.clongdouble(0.00015309874639941846 + 1j * -0.00020067404150059546),
     np.clongdouble(-6.942310483485043e-05 + 1j * 6.908980884640893e-05),
     np.clongdouble(0.00019759848369167322 + 1j * -3.021395714388174e-08),
     np.clongdouble(3.569068759934393e-05 + 1j * -2.27899159542528e-05),
     np.clongdouble(-4.532812760172859e-05 + 1j * 5.312209748107463e-05),
     np.clongdouble(4.3125356075455374e-05 + 1j * 3.49535204186948e-05),
     np.clongdouble(7.433775368550065e-05 + 1j * -4.9260859977670475e-05),
     np.clongdouble(-2.026199903598601e-05 + 1j * 1.1376881736333182e-05),
     np.clongdouble(6.968950484977909e-05 + 1j * 6.269813310424944e-07),
     np.clongdouble(2.902971811369914e-05 + 1j * -9.980698080208999e-05),
     np.clongdouble(-3.7103510624264904e-05 + 1j * 0.000122857935890825),
     np.clongdouble(-0.00010381549107198144 + 1j * -1.5743591942141152e-07),
     np.clongdouble(-4.766856883013104e-05 + 1j * 1.4529706724385748e-05)],
    [np.clongdouble(-0.0001956497863547835 + 1j * -0.00018818235993888405),
     np.clongdouble(0.00017066491748804426 + 1j * -6.633372038251998e-05),
     np.clongdouble(8.431889319263281e-05 + 1j * 0.000641812423084953),
     np.clongdouble(0.00042313304514355655 + 1j * -0.00018276307649186147),
     np.clongdouble(-0.00017119596436834984 + 1j * 0.0001710813059659968),
     np.clongdouble(-3.481077185283802e-06 + 1j * -0.0002675976708137022),
     np.clongdouble(3.3637220086788316e-05 + 1j * 0.0003053271049259059),
     np.clongdouble(0.00010648112242236894 + 1j * -0.00010131523481580286),
     np.clongdouble(-0.00022182242665703218 + 1j * -0.0003595121164884595),
     np.clongdouble(0.00019412048302525008 + 1j * -0.00016303953238125767),
     np.clongdouble(-0.0003147240909867464 + 1j * -1.8139071835896322e-05),
     np.clongdouble(0.00034209614852169837 + 1j * 0.00022347788111583795),
     np.clongdouble(-0.00038799568615575865 + 1j * 3.581660468951279e-05),
     np.clongdouble(0.00034395379027800827 + 1j * 0.00029262381482576886),
     np.clongdouble(-2.2916309215901482e-05 + 1j * 0.0002537428873026323),
     np.clongdouble(-0.0002652279245169471 + 1j * -0.00022076967985228596),
     np.clongdouble(-0.00020050025262779194 + 1j * 3.0329004651503694e-05),
     np.clongdouble(0.00010631231090102493 + 1j * 0.00024218722136049758),
     np.clongdouble(1.3363129744620445e-05 + 1j * -0.00011101745427077856),
     np.clongdouble(-4.54015993849346e-06 + 1j * 0.00026542112128109097),
     np.clongdouble(0.00025352102837850385 + 1j * -7.350265274884215e-06),
     np.clongdouble(-0.0004422452733822551 + 1j * -1.028508629173379e-05),
     np.clongdouble(0.00011061692745178322 + 1j * -0.00018151588636478055),
     np.clongdouble(0.00011187904798767028 + 1j * -0.00012535254902263252)],
    [np.clongdouble(2.8326002491501945e-05 + 1j * 6.73472532639004e-05),
     np.clongdouble(-1.3979956396227157e-05 + 1j * -4.7660082252302437e-05),
     np.clongdouble(-1.8613412634632822e-05 + 1j * -4.55338991300788e-05),
     np.clongdouble(-1.61552694272257e-05 + 1j * -0.0001177148811067739),
     np.clongdouble(-1.835151131348031e-05 + 1j * -8.674940985234944e-05),
     np.clongdouble(1.6593068732555057e-05 + 1j * 0.00029999438882493524),
     np.clongdouble(0.00015996220811603554 + 1j * -0.00034861471850842324),
     np.clongdouble(-0.0004195236977373693 + 1j * 0.0004359893748166296),
     np.clongdouble(0.0005432019852547456 + 1j * -0.0005144724284773376),
     np.clongdouble(-0.0007718155200811724 + 1j * 0.0016950264071809485),
     np.clongdouble(0.000329372917933519 + 1j * -0.0002654421699194179),
     np.clongdouble(0.00014428126795491375 + 1j * -0.0006536703536013788),
     np.clongdouble(1.9829387596111205e-05 + 1j * 3.763254056758772e-05),
     np.clongdouble(-1.2537183778935311e-05 + 1j * -0.00015638128997565612),
     np.clongdouble(-5.950237921920119e-05 + 1j * -4.2578748083829256e-05),
     np.clongdouble(9.480075120205789e-05 + 1j * 3.3558586820374926e-05),
     np.clongdouble(3.231510626261506e-05 + 1j * 0.0002555682732562677),
     np.clongdouble(4.286298784622368e-05 + 1j * 2.1660036289744334e-05),
     np.clongdouble(2.0121666976511208e-05 + 1j * 8.19465319128321e-05),
     np.clongdouble(-4.917089310978051e-06 + 1j * -0.0001726377987834528),
     np.clongdouble(-1.1721800797453063e-05 + 1j * -1.4385862255883073e-05),
     np.clongdouble(3.480997673536622e-05 + 1j * 1.3398943556066353e-05),
     np.clongdouble(-1.6351793490387904e-05 + 1j * -0.0001078693049561883),
     np.clongdouble(-4.437704994738801e-05 + 1j * -2.790609468224263e-05)]])

me2 = np.array([np.double(0.0002525903368455499),
               np.double(5.0765885218360435e-06),
               np.double(3.1291304941885141e-05),
               np.double(6.0652016857568314e-05)])


class TensorecoreMockup():

    def __init__(self):
        pass

# for( int icol = 0; icol < ncolor; icol++ )
# {
#   cxtype_sv ztemp_sv = cxzero_sv();
#   for( int jcol = 0; jcol < ncolor; jcol++ )
#     ztemp_sv += cf[icol][jcol] * jamp_sv[jcol];
#   keep = ztemp_sv;
#   deltaMEs += cxreal( ztemp_sv * cxconj( jamp_sv[icol] ) ) / denom[icol];
# }

    # re-implement org C++
    def calculate(self, jamp):
        jampr = [x.real for x in jamp]
        jampi = [x.imag for x in jamp]

        ztempr = np.matmul(cf, jampr)
        ztempi = np.matmul(cf, jampi)

        ztemp_cx = ztempr + 1j * ztempi
        jamps_cx_con = np.array(jampr) / 54 + -1j * np.array(jampi) / 54

        deltaMEs = np.matmul(ztemp_cx, jamps_cx_con)

        return deltaMEs.real

    # prepare for tensor cores
    def calculate2(self, jamp):
        # SOA for jamp
        jampri = [[x.real for x in jamp], [x.imag for x in jamp]]
        # 2d array of 0s for temp variable
        ztemp = np.zeros((2, ncolor), dtype=np.double)
        # loop over real and imag parts of jamp
        for ri in range(2):
            # go row by row through cf
            for x in range(int(ncolor / stride)):
                # 1d array of 0s for sub vector result
                subvect = np.zeros(stride, dtype=np.double)
                # loop col by col over cf
                for y in range(int(ncolor / stride)):
                    # get the submatrix out of cf
                    subcf = cf[x * stride:x * stride + stride]\
                              [:, [range(y * stride, y * stride + stride)]]
                    # remove the outer array in the result
                    subcf = [x[0] for x in subcf]
                    # get the relevant part of jamp(real/img)(vect)
                    subjamp = jampri[ri][y * stride:y * stride + stride]
                    # add sub matrix multiplication to sub vector
                    subvect += np.matmul(subcf, subjamp)
                # append sub vector result to ztemp variable
                ztemp[ri][x * stride:x * stride + stride] = subvect

        # create a real array of complex numbers
        ztemp_cx = ztemp[0] + 1j * ztemp[1]
        # create complex conjugate of jamps
        jamps_cx_con = np.array(jampri[0]) / 54 + -1j * np.array(jampri[1]) / 54

        # matrix multiplation of ztemp with compl conj of jamp
        deltaMEs = np.matmul(ztemp_cx, jamps_cx_con)
        return deltaMEs.real

    # Andrea's version
    def calculate3(self, jamp):
        jampr = [x.real for x in jamp]
        jampi = [x.imag for x in jamp]

        res1 = np.matmul(jampr, cf)
        res2 = np.matmul(res1, jampr)

        res3 = np.matmul(jampi, cf)
        res4 = np.matmul(res3, jampi)

        return (res2 + res4) / 54

    # github simplified
    def calculate4(self, jamp):
        jampr = [x.real for x in jamp]
        jampi = [x.imag for x in jamp]

        ztempr = np.matmul(cf, jampr)
        ztempi = np.matmul(cf, jampi)

        deltaME = np.matmul(ztempr, jampr) / 54 + np.matmul(ztempi, jampi) / 54
        return deltaME

    def calculate5(self, jamp):
        deltaME = np.double(0)
        for x in range(int(ncolor / stride)):
            ztemp = np.zeros((2, stride), dtype=np.double)
            for y in range(int(ncolor / stride)):
                subcf = [x[0] for x in cf[x * stride:x * stride + stride]
                          [:, [range(y * stride, y * stride + stride)]]]
                for ri in range(2):
                    subjamp = jamp[ri][y]
                    ztemp[ri] += np.matmul(subcf, subjamp)
            for ri in range(2):
                subjamp = jamp[ri][x]
                deltaME += np.matmul(subjamp, ztemp[ri])
        return deltaME / 54

    def run(self):
        for jamps, deltaME in zip(jamp2, me2):
            jampri = [[x.real for x in jamps], [x.imag for x in jamps]]
            jampri = [np.array_split(x, 3) for x in jampri]
            print('*' * 80)
            self.calculate(jamps)
            self.calculate2(jamps)
            self.calculate3(jamps)
            self.calculate4(jamps)
            res = self.calculate5(jampri)
            if deltaME == res:
                print('success')
            else:
                diff = res - deltaME
                print('result    : ' + str(res))
                print('delatME   : ' + str(deltaME))
                print('difference: ' + str(diff))
                if abs(diff) < pow(10, -19):
                    print("success within error bar" % diff)
                else:
                    print('failure')
        print('*' * 80)


if __name__ == '__main__':
    TensorecoreMockup().run()
