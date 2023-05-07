import cv2
import numpy
import time
from scipy import io
from propagation_ASM import *
import model
import tools
testpic=829
pitch=0.0036
wavelength=0.000520
n = 2160
m = 3840
z0=200
layernum=6
interval=10
reconlayer=0
slm_res = (n, m)
pad=True
convert=False
method='phase'
#method='nophase'

if pad==True:
    Hbackward = torch.complex(torch.zeros(1, layernum, 2 * n, 2 * m), torch.zeros(1, layernum, 2 * n, 2 * m))
    Hforward = torch.complex(torch.zeros(1, layernum, 2 * n, 2 * m), torch.zeros(1, layernum, 2 * n, 2 * m))
else:
    Hbackward = torch.complex(torch.zeros(1, layernum, n, m), torch.zeros(1, layernum, n, m))
    Hforward = torch.complex(torch.zeros(1, layernum,n,  m), torch.zeros(1, layernum, n, m))

Hbackward = Hbackward.cuda()
Hforward = Hforward.cuda()
for k in range(layernum):
    z=z0+k*interval
    Hbackward[:,k,:,:] = propagation_ASM2(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
    Hforward[:,k,:,:] = propagation_ASM1(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=pad,return_H=True)

net = model.codenet()
if method=='phase':
    net.load_state_dict(torch.load('phase.pth'))
if method=='nophase':
    net.load_state_dict(torch.load('nophase.pth'))
net.cuda()


init_phase=torch.zeros(1,layernum,n,m)
if method=='phase':
    for k in range(layernum):
        path = 'guessphase\\guessphase' + str(k) + '.mat'
        guessphase = io.loadmat(path)
        guessphase = guessphase['guessphase']
        init_phase[:, k, :, :] = torch.from_numpy(guessphase)

init_phase=init_phase.cuda()

imgpath='D:/zcl/python/sci3-final/testpic/amp1.png'
deppath='D:/zcl/python/sci3-final/testpic/depth.png'

input_image=tools.Loadimage(path=imgpath,channel=2,flip=0,m=m,n=n,convert=convert)
input_dep=tools.Loadimage(path=deppath,channel=2,flip=0,m=m,n=n,convert=False)
cv2.imwrite('target.png', input_image)

target_amp=torch.from_numpy(input_image)
target_amp=target_amp.view(1,1,n,m)
target_amp=torch.sqrt(target_amp)
target_amp=target_amp.cuda()
target_amp=target_amp.float()

dep=torch.from_numpy(input_dep)
dep=dep.cuda()
dep = dep.view(1, 1, n, m)
dep =dep .float()

maxdepth= torch.max(dep)
mindepth=torch.min(dep)
map =torch.zeros(1,layernum,n,m).cuda()
d = (maxdepth-mindepth) / layernum

volume =torch.zeros(1,layernum,n,m).cuda()

for k in range(layernum):
    if k==(layernum-1):
        map[:, k, :, :] = (dep >= (maxdepth - (k + 1) * d)) & (dep <= (maxdepth - k * d))
    else:
        map[:, k, :, :] = (dep >= (maxdepth - (k + 1) * d)) & (dep < (maxdepth - k * d))
    volume[:, k, :, :]  = target_amp * map

for k in range(layernum):

    z = z0 + k * interval
    target_amp_complex = torch.complex(volume[:, k, :, :] * torch.cos(init_phase[:,k,:,:] * 2 * torch.pi),
                                 volume[:, k, :, :] * torch.sin(init_phase[:,k,:,:] * 2 * torch.pi))
    target_amp_complex = target_amp_complex.view(1, 1, n, m)
    if k == 0:
        slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                    wavelength=wavelength,
                                    precomped_H=Hforward[:, k, :, :])
    else:
        slmfield += propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                     wavelength=wavelength,
                                     precomped_H=Hforward[:, k, :, :])

holo_phase = net(slmfield)
print('pass, start testing')

time_start=time.time()
with torch.no_grad():
    for kk in range(100):
        for k in range(layernum):
            z = z0 + k * interval
            target_amp_complex = torch.complex(volume[:, k, :, :] * torch.cos(init_phase[:, k, :, :] * 2 * torch.pi),
                                               volume[:, k, :, :] * torch.sin(init_phase[:, k, :, :] * 2 * torch.pi))
            target_amp_complex = target_amp_complex.view(1, 1, n, m)
            if k == 0:
                slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                            wavelength=wavelength,
                                            precomped_H=Hforward[:, k, :, :])
            else:
                slmfield += propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                             wavelength=wavelength,
                                             precomped_H=Hforward[:, k, :, :])


        holo_phase = net(slmfield)

time_end=time.time()
print('time',(time_end-time_start)/100.0)

max_phs = 2*torch.pi
holo_phase= torch.squeeze(holo_phase)
holo_phase = holo_phase -holo_phase.mean()
holophase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
cv2.imwrite('h.png', holo)


slmcomplex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
z=z0+reconlayer*interval
finalcomplex = propagation_ASM2(u_in=slmcomplex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                               wavelength=wavelength,
                               precomped_H=Hbackward[:,reconlayer,:,:])
recon_amp= torch.abs(finalcomplex)
recon_amp = torch.squeeze(recon_amp)
recon_amp=recon_amp.cpu().data.numpy()
recon=recon_amp*recon_amp
recon=tools.lin_to_srgb(recon)


recon = recon / recon.max()
pic = numpy.uint8(recon * 255)
cv2.imwrite('recon.png', pic)
