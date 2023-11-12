import numpy as np
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt

def filmCvtHSV( sbgr, type="Blue Sensitive Film Day"):
    filmSpectrum = pd.read_csv(f"../film/LvsFilm.csv")

    filmSpectrum.drop(filmSpectrum[(filmSpectrum['Wavelength'] <400)].index, inplace=True)


    spectral_area = np.sum(filmSpectrum[type].to_numpy())
    adjust = 22.6/spectral_area
    #print(spectral_area)
    filmSpectrum =adjust*filmSpectrum[type].to_numpy()/spectral_area#normalise film spectrum area.
    print(filmSpectrum)

    hsv = cv2.cvtColor(sbgr, cv2.COLOR_BGR2HSV)
    L = ( 250 / 191.25 * (255-hsv[:,:,0]))
    L_index = ((250-L)/5)

    print(f'L shape is {np.shape(L)}')
    print(np.unique(L[:,:]))
    print(f'L_index {np.unique(L_index[:,:])}')
    print(np.unique(hsv[:,:,0]))
    V = hsv[:,:,2]/255



    mono = np.zeros((sbgr.shape[0], sbgr.shape[1]))
    print(mono.shape)
    print(np.shape(filmSpectrum))
    #basisArray = basisArray*100
    for i in range(0, L.shape[0]):
        for j in range(0, L.shape[1]):
            #print(f'*** {filmSpectrum[int(L[i][j]/5)]} ****')
            if hsv[i][j][1]>3:
                mono[i][j] = V[i][j]*(filmSpectrum[int(L_index[i][j])])
            else:
                mono[i][j] = V[i][j]
            #if mono[i][j]>1:
                #mono[i][j]=1
            '''if mono[i][j] <= 0.0031308:
                mono[i][j] = 12.92*mono[i][j]
            else:
                mono[i][j] = math.pow(1.055*mono[i][j], 1/2.4) - 0.055

            if mono[i][j]>1:
                mono[i][j]=1'''

    print(f'££££££ {np.mean(mono)}')
    mono = mono/np.mean(mono)*0.5

    #med = np.median(mono)
    #mono = mono/med*127
    for i in range(0, mono.shape[0]):
        for j in range(0, mono.shape[1]):
            '''if mono[i][j] <= 0.0031308:
                mono[i][j] = 12.92*mono[i][j]
            else:
                mono[i][j] = math.pow(1.055*mono[i][j], 1/2.4) - 0.055'''

            if mono[i][j]>1:
                mono[i][j]=1


    mono = (mono*255).astype(np.uint8)

    print(np.max(mono))
    return mono


def filmCvt( sbgr, type="Blue Sensitive Film Day"):
    filmSpectrum = pd.read_csv(f"../film/LvsFilm.csv")

    filmSpectrum.drop(filmSpectrum[(filmSpectrum['Wavelength'] <360)].index, inplace=True)

    lbgrBasis = pd.read_csv("../film/lbgrMalletBasis3.csv")
    lbgrBasis.drop(lbgrBasis[(lbgrBasis['Wavelength'] <360)].index, inplace=True)

    spectral_area = np.sum(filmSpectrum[type].to_numpy())
    adjust = 22.6/spectral_area
    #print(spectral_area)
    filmSpectrum =adjust*filmSpectrum[type].to_numpy()/spectral_area#normalise film spectrum area.
    print(filmSpectrum)





    basisArray = np.vstack((lbgrBasis['B'].to_numpy(),lbgrBasis['G'].to_numpy(),lbgrBasis['R'].to_numpy())).T
    print(np.shape(basisArray))
    print(np.max(sbgr))
    print(sbgr[0][0])
    sbgr = sbgr/255.0 #Convert to range [0,1]
    print(np.max(sbgr))
    print(sbgr[0][0]*255)
    lBGR = np.zeros((sbgr.shape[0], sbgr.shape[1], sbgr.shape[2]))
    for i in range(0, sbgr.shape[0]):
        for j in range(0, sbgr.shape[1]):
            for k in range(0, sbgr.shape[2]):
                if sbgr[i][j][k] <= 0.04045:
                    lBGR[i][j][k] = sbgr[i][j][k]/12.92
                else:
                    lBGR[i][j][k] = math.pow((0.055+sbgr[i][j][k])/1.055, 2.4)

    print(np.max(lBGR))
    print(lBGR[0][0]*255)
    mono = np.zeros((sbgr.shape[0], sbgr.shape[1]))
    print(mono.shape)
    print(np.shape(filmSpectrum))
    #basisArray = basisArray*100
    for i in range(0, lBGR.shape[0]):
        for j in range(0, lBGR.shape[1]):

            mono[i][j] = ((basisArray.dot(lBGR[i][j])).T).dot(filmSpectrum)


    print(f'££££££ {np.mean(mono)}')
    mono = mono/np.mean(mono)*0.5


    for i in range(0, mono.shape[0]):
        for j in range(0, mono.shape[1]):
            if mono[i][j] <= 0.0031308:
                mono[i][j] = 12.92*mono[i][j]
            else:
                mono[i][j] = math.pow(1.055*mono[i][j], 1/2.4) - 0.055

            if mono[i][j]>1:
                mono[i][j]=1


    mono = (mono*255).astype(np.uint8)

    print(np.max(mono))
    return mono

def tvCvt( sbgr, type="Earliest Camera Tubes"):
    filmSpectrum = pd.read_csv(f"../film/LvsTVtube.csv")

    filmSpectrum.drop(filmSpectrum[(filmSpectrum['Wavelength'] <360)].index, inplace=True)

    lbgrBasis = pd.read_csv("../film/lbgrMalletBasis3.csv")
    lbgrBasis.drop(lbgrBasis[(lbgrBasis['Wavelength'] <360)].index, inplace=True)

    spectral_area = np.sum(filmSpectrum[type].to_numpy())
    adjust = 22.6/spectral_area
    #print(spectral_area)
    filmSpectrum =adjust*filmSpectrum[type].to_numpy()/spectral_area#normalise film spectrum area.
    print(filmSpectrum)

    basisArray = np.vstack((lbgrBasis['B'].to_numpy(),lbgrBasis['G'].to_numpy(),lbgrBasis['R'].to_numpy())).T
    print(np.shape(basisArray))
    print(np.max(sbgr))
    print(sbgr[0][0])
    sbgr = sbgr/255.0 #Convert to range [0,1]
    print(np.max(sbgr))
    print(sbgr[0][0]*255)
    lBGR = np.zeros((sbgr.shape[0], sbgr.shape[1], sbgr.shape[2]))
    for i in range(0, sbgr.shape[0]):
        for j in range(0, sbgr.shape[1]):
            for k in range(0, sbgr.shape[2]):
                if sbgr[i][j][k] <= 0.04045:
                    lBGR[i][j][k] = sbgr[i][j][k]/12.92
                else:
                    lBGR[i][j][k] = math.pow((0.055+sbgr[i][j][k])/1.055, 2.4)

    print(np.max(lBGR))
    print(lBGR[0][0]*255)
    mono = np.zeros((sbgr.shape[0], sbgr.shape[1]))
    print(mono.shape)
    print(np.shape(filmSpectrum))
    #basisArray = basisArray*100
    for i in range(0, lBGR.shape[0]):
        for j in range(0, lBGR.shape[1]):

            mono[i][j] = ((basisArray.dot(lBGR[i][j])).T).dot(filmSpectrum)


    print(f'££££££ {np.mean(mono)}')
    mono = mono/np.mean(mono)*0.5


    for i in range(0, mono.shape[0]):
        for j in range(0, mono.shape[1]):
            if mono[i][j] <= 0.0031308:
                mono[i][j] = 12.92*mono[i][j]
            else:
                mono[i][j] = math.pow(1.055*mono[i][j], 1/2.4) - 0.055

            if mono[i][j]>1:
                mono[i][j]=1


    mono = (mono*255).astype(np.uint8)

    print(np.max(mono))
    return mono

def lbgrBasisGraph():
    lbgrBasis = pd.read_csv("../film/lbgrMalletBasis.csv")

    fig2, ax2 = plt.subplots(figsize=(12,6))
    lbgrBasis.plot(y='B', x='Wavelength', ax=ax2, color='blue')
    lbgrBasis.plot(y='G', x='Wavelength', ax=ax2, color='green')
    lbgrBasis.plot(y='R', x='Wavelength', ax=ax2, color='red')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Reflectivity')
    ax2.set_title('Linear RGB basis functions from Mallet and Yuksel')
    ax2.grid(visible=True, which='major', axis='both')
    ax2.legend()
    ax2.set_xlim(350, 800)
    im = plt.imread('../film/rainbowSpectrum.png')
    newax = fig2.add_axes([0.125,0.11,0.775,1.0], anchor='SW', zorder=1)
    newax.imshow(im)
    newax.axis('off')
    fig2.savefig('../film/lbgrBasis.jpg')


def l_vs_film():
    filmSpectrum = pd.read_csv(f"../film/LvsFilm.csv")
    lum = pd.read_csv(f"../film/luminousity.csv")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    lum.plot(y='HVS Luminousity', x='Wavelength', ax=ax2, color='grey')
    filmSpectrum.plot(y='Blue Sensitive Film Day', x='Wavelength', ax=ax2, color='blue', alpha=0.4)
    filmSpectrum.plot(y='Orthochromatic Film Day', x='Wavelength', ax=ax2, color='green', alpha=0.4)
    filmSpectrum.plot(y='Panchromatic Film Day', x='Wavelength', ax=ax2, color='red', alpha=0.4)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Relative Sensitivity')
    ax2.set_title('Daylight Film vs Visual Luminousity of HVS')
    ax2.grid(visible=True, which='major', axis='both')
    ax2.legend()
    ax2.set_xlim([350,780])
    #ax2.set_ylim([0,0.01])
    fig2.savefig('../film/LvsFilmDay.jpg')


    fig1, ax1 = plt.subplots(figsize=(12,6))
    lum.plot(y='HVS Luminousity', x='Wavelength', ax=ax1, color='grey')
    filmSpectrum.plot(y='Blue Sensitive Film Tungsten', x='Wavelength', ax=ax1, color='blue', alpha=0.4)
    filmSpectrum.plot(y='Orthochromatic Film Tungsten', x='Wavelength', ax=ax1, color='green', alpha=0.4)
    filmSpectrum.plot(y='Panchromatic Film Tungsten', x='Wavelength', ax=ax1, color='red', alpha=0.4)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Relative Sensitivity')
    ax1.set_title('Tungsten Film vs Visual Luminousity of HVS')
    ax1.grid(visible=True, which='major', axis='both')
    ax1.legend()
    ax1.set_xlim([350,780])
    #ax2.set_ylim([0,0.01])
    fig1.savefig('../film/LvsFilmTungsten.jpg')

def l_vs_tv():
    tvSpectrum = pd.read_csv(f"../film/LvsTVTube.csv")
    lum = pd.read_csv(f"../film/luminousity.csv")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    lum.plot(y='HVS Luminousity', x='Wavelength', ax=ax2, color='black')
    tvSpectrum.plot(y='Earliest Camera Tubes', x='Wavelength', ax=ax2, color='red', alpha=0.7)
    tvSpectrum.plot(y='Old CPS Emitron & PES Photicon', x='Wavelength', ax=ax2, color='blue', alpha=0.7)
    tvSpectrum.plot(y='Tri-Alkali CPS Emitron', x='Wavelength', ax=ax2, color='magenta', alpha=0.7)
    tvSpectrum.plot(y='Standard Emitron', x='Wavelength', ax=ax2, color='orange', alpha=0.7)

    tvSpectrum.plot(y='Image Orthicon', x='Wavelength', ax=ax2, color='green', alpha=0.7)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Relative Sensitivity')
    ax2.set_title('TV Tube Sensitivity vs Visual Luminousity of HVS')
    ax2.grid(visible=True, which='major', axis='both')
    ax2.legend()
    ax2.set_xlim([350,780])
    ax2.set_ylim([0,1.1])
    fig2.savefig('../film/LvsTVTube.jpg')




#image_name='../film/blue.jpg'
#image_name='../film/RAF_type_A1_roundel.jpg'
#image_name='ModernHawker.png'
#image_name='hawkerSynth.jpg'
image_name='./bgr.png'
#image_name='Hawker3.png'
#image_name='015004_gt.jpg'
bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
Lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L = Lab[:,:,0]
L = cv2.merge((L,L,L))
print(L.shape)

graph = cv2.imread('../film/HVSSensitivity.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)

all = cv2.hconcat((graph, L,bgr))
cv2.imwrite('../film/AllOrig.jpg', all)
all = cv2.hconcat(( L,bgr))
cv2.imwrite('../film/AllOrigWOGraph.jpg', all)
cv2.imwrite('../film/L-channel.jpg',L)

#bgr = cv2.imread(image_name)
type = "Blue Sensitive Film Day"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/BlueSensitiveSensDay.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllBlueDay.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Orthochromatic Film Day"
mono = filmCvt(bgr, type=type)
graph = cv2.imread('../film/OrthochromaticSensDay.jpg')
graph = cv2.resize(graph,(600,600))
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllOrthoDay.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Panchromatic Film Day"
mono = filmCvt(bgr, type=type)
graph = cv2.imread('../film/PanchromaticSensDay.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllPanDay.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Blue Sensitive Film Tungsten"
mono = filmCvt(bgr, type=type)
graph = cv2.imread('../film/BlueSensitiveSensTungsten.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllBlueTung.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Orthochromatic Film Tungsten"
mono = filmCvt(bgr, type=type)
graph = cv2.imread('../film/OrthochromaticSensTungsten.jpg')
graph = cv2.resize(graph,(600,600))
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllOrthoTung.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Panchromatic Film Tungsten"
mono = filmCvt(bgr, type=type)
graph = cv2.imread('../film/PanchromaticSensTungsten.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllPanTung.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

bgr = cv2.imread(image_name)
#bgr = cv2.resize(bgr,(600,600))
type = "Y"
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
Y = yuv[:,:,0]
Lab[:,:,0] = Y
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
Y = cv2.merge((Y,Y,Y))
all = cv2.hconcat((Y, bgr))
cv2.imwrite('../film/AllY.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

'''
type = "IlfordOrtho80"
mono = filmCvt(bgr, type=type)

cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


type = "ADOXCMS20"
mono = filmCvt(bgr, type=type)

cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


type = "IlfordDelta400-3200"
mono = filmCvt(bgr, type=type)

cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
'''

bgr = cv2.imread(image_name)
type = "Earliest Camera Tubes"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/EarliestCameraTubes.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllEarliestCameraTubes.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Image Orthicon"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/ImageOrthicon.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllImageOrthicon.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Standard Emitron"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/StandardEmitron.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllStandardEmitron.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)


bgr = cv2.imread(image_name)
type = "Old CPS Emitron & PES Photicon"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/OldCPSEmitronPESPhoticon.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllOldCPSEmitronPESPhoticon.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)



bgr = cv2.imread(image_name)
type = "Tri-Alkali CPS Emitron"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
graph = cv2.imread('../film/Tri-AlkaliCPSEmitron.jpg')
graph = cv2.resize(graph,(600,600))
print(graph.shape)
Lab[:,:,0] = mono
bgr = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
mono = cv2.merge((mono,mono,mono))
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)
print('****************')
print(bgr.shape)
print(mono.shape)
print(graph.shape)
print('****************')
all = cv2.hconcat((graph, mono, bgr))
cv2.imwrite('../film/AllTri-AlkaliCPSEmitron.jpg', all)
cv2.imwrite(f'../film/{image_name}_{type}.png', mono)

lbgrBasisGraph()
l_vs_film()
l_vs_tv()



