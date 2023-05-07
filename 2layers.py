import cv2
import numpy
from scipy import io
from propagation_ASM import *
import model
import tools
pitch=0.0036
wavelength=0.000520
n = 2160
m = 3840
z0=160 #juli
layernum=6
interval=10
layer1=0
layer2=3
reconlayer=layer2
slm_res = (n, m)
pad=True
convert=False
method='phase'
#method='nophase'
img1path='D:/zcl/python/sci3-final/testpic/ttt1.png'
img2path='D:/zcl/python/sci3-final/testpic/ttt2.png'
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



input_image1=tools.Loadimage(path=img1path,channel=2,flip=0,m=m,n=n,convert=convert)
input_image2=tools.Loadimage(path=img2path,channel=1,flip=0,m=m,n=n,convert=convert)

target_amp1=torch.from_numpy(input_image1)
target_amp1=target_amp1.view(1,1,n,m)
target_amp1=torch.sqrt(target_amp1)
target_amp1=target_amp1.cuda()
target_amp1=target_amp1.float()

target_amp2=torch.from_numpy(input_image2)
target_amp2=target_amp2.view(1,1,n,m)
target_amp2=torch.sqrt(target_amp2)
target_amp2=target_amp2.cuda()
target_amp2=target_amp2.float()


z=z0+layer1*interval
target_amp_complex = torch.complex(target_amp1 * torch.cos(init_phase[:, layer1, :, :] * 2 * torch.pi),
                                   target_amp1* torch.sin(init_phase[:,layer1, :, :] * 2 * torch.pi))
target_amp_complex = target_amp_complex.view(1, 1, n, m)

slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                            wavelength=wavelength,
                            precomped_H=Hforward[:, layer1, :, :])

z=z0+layer2*interval
target_amp_complex = torch.complex(target_amp2 * torch.cos(init_phase[:, layer2, :, :] * 2 * torch.pi),
                                   target_amp2 * torch.sin(init_phase[:,layer2, :, :] * 2 * torch.pi))
target_amp_complex = target_amp_complex.view(1, 1, n, m)

slmfield += propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                            wavelength=wavelength,
                            precomped_H=Hforward[:, layer2, :, :])


holo_phase = net(slmfield)
print('pass')

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