#  Abstract：2D Microstructure-based 3D Random Porous Materials Plot in Mayavi
#  Author：Yutai Su (buaa_syt@126.com)
#  Edited by: PyCharm Edu
#  Date：2023-3-18
#  Introduction：Generate and plot 3D random porous structure

# 0.1 Import packages
import matplotlib.pyplot as plt
from mayavi import mlab
import scipy.ndimage as spn
from scipy import fftpack as sp_ft
import cv2
import numpy as np
from scipy.optimize import minimize
import os

# 0.2 Predefined functions
# 0.2.1 Preprocess the 2D Picture (from RGB to binary)
def rgb2bi(rgb, phaserange=[0.0, 0.7]):
    """
    Get the binary matrix from RGB matrix
    :param rgb: 2D RGB matrix
    :param phaserange: gray value range of target phase
    :return: bina is 2D binary matrix, phasenumber is the number array of target phase in the 2D matrix
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r / 255 + 0.5870 * g / 255 + 0.1140 * b / 255
    bina = np.zeros(gray.shape)
    bina[gray < phaserange[0]] = 1
    bina[gray > phaserange[1]] = 1
    xx, yy = np.where(bina == 1)
    phasenumber = 1+xx+yy*bina.shape[0]
    return bina, phasenumber

# 0.2.2 Two-point-probability-function for 2D or 3D binary matrix
def TwoPoint2D_3D(img, nbins=500, r_max=100):
    """
    Get the Two-point-probability-function for 2D or 3D binary matrix
    :param img: 2D or 3D binary matrix
    :param nbins:
    :param r_max:
    :return: distance-probability data of the two-point-probability-function
    """
    # Calculate half lengths of the image
    hls = (np.ceil(np.shape(img)) / 2).astype(int)
    # Fourier Transform and shift image
    F = sp_ft.ifftshift(sp_ft.fftn(sp_ft.fftshift(img)))
    # Compute Power Spectrum
    P = np.absolute(F**2)
    # Auto-correlation is inverse of Power Spectrum
    autoc = np.absolute(sp_ft.ifftshift(sp_ft.ifftn(sp_ft.fftshift(P))))
    if len(autoc.shape) == 2:
        # obtain dt is the distance in the img
        adj = np.reshape(autoc.shape, [2, 1, 1])
        inds = np.indices(autoc.shape) - adj / 2
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autoc.shape) == 3:
        adj = np.reshape(autoc.shape, [3, 1, 1, 1])
        inds = np.indices(autoc.shape) - adj / 2
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')
    bin_size = int(np.ceil(r_max / nbins))
    bins = np.arange(bin_size, r_max, step=bin_size)
    radial_sum = np.zeros_like(bins)
    # print(bins)
    for i, r in enumerate(bins):
        # Generate Radial Mask from dt using bins
        mask = (dt <= r) * (dt > (r - bin_size))
        radial_sum[i] = np.sum(autoc[mask]) / np.sum(mask)
    # Return normalized bin and radially summed autoc
    norm_autoc_radial = radial_sum / np.max(autoc)
    tpcf_distance = bins
    tpcf_probability = norm_autoc_radial
    tpcf_probability = np.insert(tpcf_probability, 0, 1)
    tpcf_distance = np.append(tpcf_distance, 2*tpcf_distance[len(tpcf_distance)-1]-tpcf_distance[len(tpcf_distance)-2])
    return tpcf_distance, tpcf_probability

# 0.2.3 Two-point-probability-function for 2D or 3D binary matrix
def RPS_3D_MAPE(Length, volfra, pro0, test_img):
    """
    MAPE value for the generated 3D microstructual matrix
    :param Length:
    :param volfra:
    :param pro0:
    :param test_img:
    :return:
    """
    SD = Length/2
    Processed_Img = spn.filters.gaussian_filter(test_img, SD)
    PerLimit = np.percentile(Processed_Img.flatten(), volfra*100)
    # 4. Identifier for two phase materials(Pore phase and solid phase)
    Identifier = Processed_Img.copy()
    Identifier[Processed_Img >= PerLimit] = 0
    Identifier[Processed_Img < PerLimit] = 1
    dis1, pro1 = TwoPoint2D_3D(Identifier, r_max=45)
    iMax = min(len(pro0), len(pro1))
    F_sum = np.sum(pro0[0:iMax])
    F_abs = np.sum(np.abs(pro0[0:iMax]-pro1[0:iMax]))
    MAPE = F_abs/F_sum
    return MAPE, Identifier, pro1

# 1.0 Initialization of 2D microstructure-based parameters
# imgdir: input location of 2D picture
imgdir = "1.png"
# phaserange: gray value range of target phase
phaserange = [0.0, 0.0025]
# oripixlen: the original length of one pixel in micro-meter
oripixlen = 10.0/290.0
# tarpixlen: the pixel number in target 3D structure
SampleSize = 50
# tarpixlen: the target length of one pixel in micro-meter
tarpixlen = 10.0/SampleSize
# r_maxs: the maximun distance of the two-point-probability-function
r_maxs = 45
# resize the original picture to match the target 3D RVE
imgorig = plt.imread(imgdir)
newsize = (int(imgorig.shape[1]/tarpixlen*oripixlen),
           int(imgorig.shape[0]/tarpixlen*oripixlen))
img_pro = cv2.resize(cv2.GaussianBlur(imgorig, (5, 5), 0),
                     newsize, interpolation=cv2.INTER_AREA)
img, inform = rgb2bi(img_pro, phaserange)
# volfra: the volume fraction of target phase
volfra = len(inform)/img.shape[0]/img.shape[1]
print(volfra)
# dis, pro: the distance and probability values of the two-point-probability-function
dis, pro = TwoPoint2D_3D(img, r_max=r_maxs)

# 2.0 Parameter definition of 3D random porous structure
randomseed = 1000
Shape = [SampleSize, SampleSize, SampleSize]
PixelNum = np.prod(Shape)
np.random.seed(randomseed)
rand_img = np.random.standard_normal(Shape)

# 3 MAPE optimization for the target two-point-probability-function
fun = lambda x1: (RPS_3D_MAPE(x1[0], volfra, pro, rand_img)[0])
res = minimize(fun, [1], method='nelder-mead', options={'maxiter': 100, 'xatol': 1e-8, 'disp': True})
length = res.x[0]
MAPE, Identifier, pro1 = RPS_3D_MAPE(length, volfra, pro, rand_img)

# 4 Plot 3D RVE structures
fig = mlab.figure(bgcolor=(1, 1, 1))
mlab.outline(color=(0, 0, 0), line_width=1.2)
# 4.1 Solid phase Coordinates (Identifier is 1)
xx, yy, zz = np.where(Identifier == 1)
# 4.2 Target phase Coordinates (Identifier is 0)
xx1, yy1, zz1 = np.where(Identifier == 0)
# 4.3 Solid phase plot
mlab.points3d(xx, yy, zz,
              mode="cube",
              color=(0.9, 0.9, 0.9),
              line_width=0,
              opacity=1.0,
              scale_factor=1)
# 4.4 targe phase plot
mlab.points3d(xx1, yy1, zz1,
              mode="cube",
              color=(0.0, 0.0, 0.0),
              line_width=0,
              opacity=0.2,
              scale_factor=1)
mlab.view(azimuth=40.0, elevation=70.0, distance=SampleSize*6, focalpoint=np.array([SampleSize/2-0.5, SampleSize/2-0.5, SampleSize/2-0.5]))
mlab.show()

# 5 Save the inp files
filedir = "inpfiles"
modname = "mode1"
if os.path.exists(filedir):
    InpPF = open("".join([filedir, "/", modname, "_sets.inp"]), 'w')
else:
    os.makedirs(filedir)
    InpPF = open("".join([filedir, "/", modname, "_sets.inp"]), 'w')
InpPF.write('** ----------------------------------------------------------------')
InpPF.write('\n')
InpPF.write('** Yutai Su (buaa_syt@126.com)')
InpPF.write('\n')
InpPF.write('** Generated by Python')
InpPF.write('\n')
InpPF.write('** ----------------------------------------------------------------')
InpPF.write('\n')
InpPF.write('*Elset, elset=Solid_Phase')
InpPF.write('\n')
SolidElemNumSet = np.unique((1-Identifier.flatten())*range(1, len(Identifier.flatten())+1))
SolidElemNumSet = SolidElemNumSet[SolidElemNumSet != 0].astype(int)
i = 0
while i < len(SolidElemNumSet):
    k = 0
    while k < 16:
        InpPF.write(' ')
        if i < len(SolidElemNumSet):
            InpPF.write(str(SolidElemNumSet[i]))
            InpPF.write(',')
        else:
            break
        i += 1
        k += 1
    InpPF.write('\n')
InpPF.write('*Elset, elset=Target_Phase')
InpPF.write('\n')
TarElemNumSet = np.unique(Identifier.flatten()*range(1, len(Identifier.flatten())+1))
PoreElemNumSet = TarElemNumSet[TarElemNumSet != 0].astype(int)
i = 0
while i < len(SolidElemNumSet):
    k = 0
    while k < 16:
        InpPF.write(' ')
        if i < len(SolidElemNumSet):
            InpPF.write(str(SolidElemNumSet[i]))
            InpPF.write(',')
        else:
            break
        i += 1
        k += 1
    InpPF.write('\n')
InpPF.close()
