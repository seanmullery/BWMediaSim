import numpy as np
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt


def filmCvt( sbgr, type="Blue Sensitive Film Day"):
    filmSpectrum = pd.read_csv(f"./MediaSpecResponse/LvsFilm.csv")

    filmSpectrum.drop(filmSpectrum[(filmSpectrum['Wavelength'] <360)].index, inplace=True)

    lbgrBasis = pd.read_csv("./MediaSpecResponse/lbgrMulleryBasis.csv")
    lbgrBasis.drop(lbgrBasis[(lbgrBasis['Wavelength'] <360)].index, inplace=True)

    spectral_area = np.sum(filmSpectrum[type].to_numpy())
    adjust = 22.6/spectral_area

    filmSpectrum =adjust*filmSpectrum[type].to_numpy()/spectral_area #normalise film spectrum area.

    basisArray = np.vstack((lbgrBasis['B'].to_numpy(),lbgrBasis['G'].to_numpy(),lbgrBasis['R'].to_numpy())).T

    sbgr = sbgr/255.0 #Convert to range [0,1]

    lBGR = np.zeros((sbgr.shape[0], sbgr.shape[1], sbgr.shape[2]))
    for i in range(0, sbgr.shape[0]):
        for j in range(0, sbgr.shape[1]):
            for k in range(0, sbgr.shape[2]):
                if sbgr[i][j][k] <= 0.04045:
                    lBGR[i][j][k] = sbgr[i][j][k]/12.92
                else:
                    lBGR[i][j][k] = math.pow((0.055+sbgr[i][j][k])/1.055, 2.4)


    mono = np.zeros((sbgr.shape[0], sbgr.shape[1]))

    for i in range(0, lBGR.shape[0]):
        for j in range(0, lBGR.shape[1]):

            mono[i][j] = ((basisArray.dot(lBGR[i][j])).T).dot(filmSpectrum)



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


    return mono

def tvCvt( sbgr, type="Earliest Camera Tubes"):
    filmSpectrum = pd.read_csv(f"./MediaSpecResponse/LvsTVtube.csv")

    filmSpectrum.drop(filmSpectrum[(filmSpectrum['Wavelength'] <360)].index, inplace=True)

    lbgrBasis = pd.read_csv("./MediaSpecResponse/lbgrMulleryBasis.csv")
    lbgrBasis.drop(lbgrBasis[(lbgrBasis['Wavelength'] <360)].index, inplace=True)

    spectral_area = np.sum(filmSpectrum[type].to_numpy())
    adjust = 22.6/spectral_area

    filmSpectrum =adjust*filmSpectrum[type].to_numpy()/spectral_area#normalise film spectrum area.

    basisArray = np.vstack((lbgrBasis['B'].to_numpy(),lbgrBasis['G'].to_numpy(),lbgrBasis['R'].to_numpy())).T

    sbgr = sbgr/255.0 #Convert to range [0,1]

    lBGR = np.zeros((sbgr.shape[0], sbgr.shape[1], sbgr.shape[2]))
    for i in range(0, sbgr.shape[0]):
        for j in range(0, sbgr.shape[1]):
            for k in range(0, sbgr.shape[2]):
                if sbgr[i][j][k] <= 0.04045:
                    lBGR[i][j][k] = sbgr[i][j][k]/12.92
                else:
                    lBGR[i][j][k] = math.pow((0.055+sbgr[i][j][k])/1.055, 2.4)


    mono = np.zeros((sbgr.shape[0], sbgr.shape[1]))


    for i in range(0, lBGR.shape[0]):
        for j in range(0, lBGR.shape[1]):

            mono[i][j] = ((basisArray.dot(lBGR[i][j])).T).dot(filmSpectrum)



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
    return mono

image_name='./bgr.png'

bgr = cv2.imread(image_name)

Lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L = Lab[:,:,0]

cv2.imwrite('./OutputMedia/L-channel.jpg',L)


type = "Blue Sensitive Film Day"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)


type = "Orthochromatic Film Day"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)


bgr = cv2.imread(image_name)
type = "Panchromatic Film Day"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Blue Sensitive Film Tungsten"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)


bgr = cv2.imread(image_name)
type = "Orthochromatic Film Tungsten"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)


bgr = cv2.imread(image_name)
type = "Panchromatic Film Tungsten"
mono = filmCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Y"
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
Y = yuv[:,:,0]
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)



bgr = cv2.imread(image_name)
type = "Earliest Camera Tubes"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Image Orthicon"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Standard Emitron"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'../OutputMedia/{image_name[2:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Old CPS Emitron & PES Photicon"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)

bgr = cv2.imread(image_name)
type = "Tri-Alkali CPS Emitron"
mono = tvCvt(bgr, type=type)
cv2.imwrite(f'./OutputMedia/{image_name[2:-4]}_{type}.png', mono)



