# Plot loss results from digit_recog_CNN.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

plotX = 'lambdaR'

my_file = Path("Max_MI_VERSION_NOT_IMPLEMENTED_relu_trstep100.txt")
print(my_file)
# Check file existence
if my_file.is_file():
    # Read file
    my_file = str(my_file)
    crs = open(my_file, "r")
    trstepF = []
    lambdaF = []
    nF = []
    trstepR = []
    lambdaR = []
    nR = []
    maxY = []
    corrX = []
    print('WE ARE STUDYING THE MI OF THE LAST LAYER')

    for line in crs:
        trstepF.append(line.split()[0])
        lambdaF.append(line.split()[1])
        nF.append(line.split()[2])
        trstepR.append(line.split()[3])
        lambdaR.append(line.split()[4])
        nR.append(line.split()[5])
        maxY.append(line.split()[16])
        corrX.append(line.split()[17])
    crs.close()

    # Eliminate heading row (python 2.7)
    trstepF = map(float,trstepF[1:])
    lambdaF = map(float,lambdaF[1:])
    nF = map(float,nF[1:])
    trstepR = map(float,trstepR[1:])
    lambdaR = map(float,lambdaR[1:])
    nR = map(float,nR[1:])
    maxY = map(float,maxY[1:])
    corrX = map(float,corrX[1:])

    create = 1
    if plotX == 'lambdaR':
        for ii in range(0,len(lambdaR)):
            if lambdaF[ii] == 0.0:
                if create == 1:
                    xvec = []
                    xvec.append(lambdaR[ii])
                    yvec = []
                    yvec.append(maxY[ii])
                    varvec = []
                    varvec.append(maxY[ii]*maxY[ii])
                    ycount = []
                    ycount.append(1)
                    create = 0
                else:
                    rep = 0
                    for jj in range(0,len(xvec)):
                        if xvec[jj] == lambdaR[ii]:
                            rep = 1
                            yvec[jj] = yvec[jj]+maxY[ii]
                            varvec[jj] = varvec[jj]+maxY[ii]*maxY[ii]
                            ycount[jj] = ycount[jj]+1
                    if rep == 0:
                            xvec.append(lambdaR[ii])
                            yvec.append(maxY[ii])
                            varvec.append(maxY[ii]*maxY[ii])
                            ycount.append(1)
        for jj in range(0,len(yvec)):
            yvec[jj] = yvec[jj]/ycount[jj]
            varvec[jj] = varvec[jj]/ycount[jj]
        varvec = [x-y for (x,y) in zip(varvec,[ii ** 2 for ii in yvec])]
        stdvec = [ii ** 0.5 for ii in varvec]

    if plotX == 'lambdaF':
        for ii in range(0,len(lambdaF)):
            if lambdaR[ii] == 0.0:
                if create == 1:
                    xvec = []
                    xvec.append(lambdaF[ii])
                    yvec = []
                    yvec.append(maxY[ii])
                    varvec = []
                    varvec.append(maxY[ii]*maxY[ii])
                    ycount = []
                    ycount.append(1)
                    create = 0
                else:
                    rep = 0
                    for jj in range(0,len(xvec)):
                        if xvec[jj] == lambdaF[ii]:
                            rep = 1
                            yvec[jj] = yvec[jj]+maxY[ii]
                            varvec[jj] = varvec[jj]+maxY[ii]*maxY[ii]
                            ycount[jj] = ycount[jj]+1
                    if rep == 0:
                            xvec.append(lambdaF[ii])
                            yvec.append(maxY[ii])
                            varvec.append(maxY[ii]*maxY[ii])
                            ycount.append(1)
        for jj in range(0,len(yvec)):
            yvec[jj] = yvec[jj]/ycount[jj]
            varvec[jj] = varvec[jj]/ycount[jj]
        varvec = [x-y for (x,y) in zip(varvec,[ii ** 2 for ii in yvec])]
        stdvec = [ii ** 0.5 for ii in varvec]

    print(xvec)
    print('Mean:')
    print(yvec)
    print('Std:')
    print(stdvec)
    print(ycount)

    #Sort
    sortvec = sorted(range(len(xvec)), key=lambda k: xvec[k])
    xvecsort = []
    yvecsort = []
    ycountsort = []
    for ii in range(0,len(xvec)):
        xvecsort.append(xvec[sortvec[ii]])
        yvecsort.append(yvec[sortvec[ii]])
        ycountsort.append(ycount[sortvec[ii]])

    xvec = xvecsort
    yvec = yvecsort
    ycount = ycountsort
    print(xvec)
    print('Mean:')
    print(yvec)
    print('Std:')
    print(stdvec)
    print(ycount)

    plt.figure(1)
    plt.errorbar(xvec,yvec,yerr=stdvec,linestyle='None',marker='o')
    plt.xlabel(plotX)
    plt.ylabel('Maximum I(Y;T)')
    plt.gca().set_xlim([xvec[1]/10,xvec[-1]*10])
    #plt.legend(loc=2)
    plt.xscale('log')
    #plt.yscale('log')
    #plt.grid(b=True,which='major')
    #plt.minorticks_on
    #plt.grid(b=True,which='minor')
    plt.hlines(y=yvec[0]+stdvec[0],xmin=xvec[1]/10,xmax=xvec[-1]*10,linestyles='dashed',alpha=0.3)
    plt.hlines(y=yvec[0]-stdvec[0],xmin=xvec[1]/10,xmax=xvec[-1]*10,linestyles='dashed',alpha=0.3)
    plt.hlines(y=yvec[0],xmin=xvec[1]/10,xmax=xvec[-1]*10,linestyles='dashed')
else:
    print('ERROR: file not found.')


plt.show()
