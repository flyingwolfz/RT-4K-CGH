import cv2
import numpy
import time
from skimage.metrics import structural_similarity as ssim
from propagation_ASM import *
import model
import tools
from scipy import io

imgpath='E:/DIV2K/DIV2K_valid_HR/0887.png'
pitch=0.0036
wavelength=0.000520
n = 2160
m = 3840
z0=160 #juli
layernum=6
interval=10
testlayer=1

slm_res = (n, m)
pad=True
convert=False
method='phase'
#method='nophase'

z=z0+testlayer*interval
Hbackward= propagation_ASM2(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                            wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
Hforward = propagation_ASM1(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                            wavelength=wavelength, z=z, linear_conv=pad,return_H=True)
Hbackward = Hbackward.cuda()
Hforward = Hforward.cuda()

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

net = model.codenet()
if method=='phase':
    net.load_state_dict(torch.load('phase.pth'))
if method=='nophase':
    net.load_state_dict(torch.load('nophase.pth'))
net.cuda()

input_image=tools.Loadimage(path=imgpath,channel=2,flip=0,m=m,n=n,convert=convert)
if convert == True:
    input_image2  = tools.lin_to_srgb(input_image)
else:
    input_image2=input_image
input_image2= numpy.uint8(input_image2 * 255)
cv2.imwrite('target.png', input_image2)
target_amp = torch.from_numpy(input_image)
target_amp = target_amp.cuda()
target_amp = torch.sqrt(target_amp)
target_amp = target_amp.view(1, 1, n, m)
target_amp = target_amp.float()

init_phase=torch.zeros(1,1,n,m)

if method=='phase':
    path = 'guessphase\\guessphase' + str(testlayer) + '.mat'
    phase = io.loadmat(path)
    phase = phase['guessphase']
    init_phase = torch.from_numpy(phase)

init_phase=init_phase.cuda()

target_amp_complex = torch.complex(target_amp * torch.cos(init_phase * 2 * torch.pi),
                            target_amp * torch.sin(init_phase * 2 * torch.pi))

slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                        wavelength=wavelength,
                        precomped_H=Hforward)
with torch.no_grad():
   holo_phase = net(slmfield)
print('pass, start testing')

time_start=time.time()
with torch.no_grad():
    for k in range(100):
        target_amp_complex = torch.complex(target_amp * torch.cos(init_phase * 2 * torch.pi),
                                           target_amp * torch.sin(init_phase * 2 * torch.pi))
        slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                    wavelength=wavelength,
                                    precomped_H=Hforward)
        holo_phase = net(slmfield)
time_end=time.time()
print('time',(time_end-time_start)/100.0)

slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
recon_complex = propagation_ASM2(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                wavelength=wavelength,
                                precomped_H=Hbackward)

recon_amp = torch.abs(recon_complex)

recon_amp = torch.squeeze(recon_amp)
recon_amp=recon_amp.cpu().data.numpy()
recon=recon_amp*recon_amp
if convert == True:
    recon = tools.lin_to_srgb(recon)
target_amp=torch.squeeze(target_amp)
target_amp = target_amp.cpu().numpy()
psnrr = psnr(recon_amp, target_amp)
print('psnr:',psnrr)
ssimm = ssim(recon_amp, target_amp)
print('ssim:',ssimm)

recon = recon / recon.max()
pic = numpy.uint8(recon * 255)
cv2.imwrite('recon.png', pic)

max_phs = 2 * torch.pi
holo_phase = torch.squeeze(holo_phase)
#holo_phase = holo_phase -holo_phase.mean()
holophase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
holo = numpy.uint8(holophase.cpu().data.numpy() * 255)
cv2.imwrite('h.png', holo)

max_phs = 1
phasepic = torch.squeeze(init_phase)
phasepic = ((phasepic + max_phs / 2) % max_phs) / max_phs
pic = numpy.uint8(phasepic.cpu().data.numpy() * 255)
cv2.imwrite('predict_phase.png', pic)