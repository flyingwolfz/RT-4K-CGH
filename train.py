from torch import nn
import cv2
import numpy
from tqdm import trange
import time
from scipy import io
import model
from propagation_ASM import *
import tools
torch.cuda.manual_seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed_all(12)
num=50
trainnum=700
validnum=100
pitch=0.0036
wavelength=0.000520
n = 2160
m = 3840
z0=160
layernum=6
interval=10
slm_res = (n, m)
pad=True
convert=False
method='phase'
#method='nophase'

if pad==True:
    Hbackward=torch.complex(torch.zeros(1,layernum,2*n, 2*m), torch.zeros(1,layernum,2*n, 2*m))
    Hforward=torch.complex(torch.zeros(1,layernum,2*n, 2*m), torch.zeros(1,layernum,2*n, 2*m))
else:
    Hbackward = torch.complex(torch.zeros(1, layernum,  n, m), torch.zeros(1, layernum, n, m))
    Hforward = torch.complex(torch.zeros(1, layernum, n, m), torch.zeros(1, layernum, n, m))

for k in range(layernum):
    z=z0+k*interval
    Hbackward[:,k,:,:] = propagation_ASM2(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
    Hforward[:,k,:,:] = propagation_ASM1(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=pad,return_H=True)

Hbackward = Hbackward.cuda()
Hforward = Hforward.cuda()
lr=0.001

net = model.codenet()

criterion = nn.MSELoss()
net=net.cuda()
criterion=criterion.cuda()
init_phase=torch.zeros(1,layernum,n,m)
init_phase=init_phase.cuda()

optvars = [{'params': net.parameters()}]

if method=='phase':
    init_phase = init_phase.requires_grad_(True)
    optvars += [{'params': init_phase}]

optimizier = torch.optim.Adam(optvars, lr=lr)

trainpath='D:\\DIV2K_train_HR'
validpath='D:\\DIV2K_valid_HR'

for k in trange(num):
    currenttloss = 0
    currentloss = 0
    for kk in range(trainnum):

        optimizier.zero_grad()

        currentlayer = numpy.random.randint(low=0, high=layernum)

        image_index = 100 + kk
        flip = numpy.random.randint(low=0, high=100)
        target_amp=tools.loadimage(path=trainpath,image_index=image_index,channel=2,flip=flip,m=m,n=n,convert=convert)

        real = torch.cos(init_phase[:,currentlayer, :, :] * 2 * torch.pi)
        imag = torch.sin(init_phase[:,currentlayer, :, :]  * 2 * torch.pi)
        target_amp_complex = torch.complex(target_amp*real, target_amp*imag)

        slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                    wavelength=wavelength,
                                    precomped_H=Hforward[:, currentlayer, :, :])
        holo_phase = net(slmfield)

        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        recon_complex = propagation_ASM2(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                        wavelength=wavelength,
                                        precomped_H=Hbackward[:,currentlayer, :, :])

        recon_amp = torch.abs(recon_complex)
        loss = criterion(recon_amp,target_amp)

        currenttloss = currenttloss + loss.cpu().data.numpy()
        loss.backward()
        optimizier.step()

    print('trainloss:', currenttloss / trainnum)


    with torch.no_grad():
        for kk in range(validnum):
            currentlayer = numpy.random.randint(low=0, high=layernum)
            image_index = 801 + kk
            flip = numpy.random.randint(low=0, high=100)

            target_amp = tools.loadimage(path=validpath, image_index=image_index, channel=2, flip=flip, m=m, n=n,
                                         convert=convert)

            real = torch.cos(init_phase[:, currentlayer, :, :] * 2 * torch.pi)
            imag = torch.sin(init_phase[:, currentlayer, :, :] * 2 * torch.pi)
            target_amp_complex = torch.complex(target_amp * real, target_amp * imag)

            slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                       wavelength=wavelength,
                                       precomped_H=Hforward[:, currentlayer, :, :])
            holo_phase = net(slmfield)

            slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
            recon_complex = propagation_ASM2(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                            wavelength=wavelength,
                                            precomped_H=Hbackward[:, currentlayer, :, :])

            recon_amp = torch.abs(recon_complex)
            loss = criterion(recon_amp, target_amp)
            currentloss = currentloss + loss.cpu().data.numpy()

            if  k%10==0 and kk == 38:
                c = k
                max_phs = 2 * torch.pi
                holo_phase = torch.squeeze(holo_phase)
                #holophase = output - output.mean()
                holophase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
                holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
                b = 'h\\' + method+str(c)
                imgpath = b + '.png'
                cv2.imwrite(imgpath, holo)

        print('validloss:', currentloss / validnum)
    time.sleep(1)
pass

pthname=method+'.pth'
torch.save(net.state_dict(), pthname)

if method=='phase':
    for kk in range(layernum):
        c = kk
        path = 'guessphase\\guessphase' + str(c) + '.mat'
        guessphase = torch.squeeze(init_phase[:, kk, :, :])
        guessphase = guessphase.cpu().data.numpy()
        guessphasemat = numpy.mat(guessphase)
        io.savemat(path, {'guessphase': guessphasemat})
