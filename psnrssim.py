import numpy
from skimage.metrics import structural_similarity as ssim
import model
from propagation_ASM import *
import tools
from scipy import io
def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

pitch=0.0036
wavelength=0.000520
n = 2160
m = 3840
z0=160 #juli
layernum=6
interval=10
testlayer=5

slm_res = (n, m)
pad=True
convert=False
method='phase'
#method='nophase'

z=z0+testlayer*interval
Hbackward = propagation_ASM2(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=-z, linear_conv=pad,return_H=True)
Hbackward = Hbackward.cuda()

Hforward = propagation_ASM1(torch.empty(1, 1, n, m), feature_size=[pitch, pitch],
                                wavelength=wavelength, z=z, linear_conv=pad,return_H=True)
Hforward = Hforward.cuda()

net = model.codenet()
if method=='phase':
    net.load_state_dict(torch.load('phase.pth'))
if method=='nophase':
    net.load_state_dict(torch.load('nophase.pth'))
net.cuda()

validpath='D:\\DIV2K_valid_HR'

init_phase=torch.zeros(1,1,n,m)

if method=='phase':
    path = 'guessphase\\guessphase' + str(testlayer) + '.mat'
    phase = io.loadmat(path)
    phase = phase['guessphase']
    init_phase = torch.from_numpy(phase)

init_phase=init_phase.cuda()

rangege=100
currentssim=0
currentpsnr=0
with torch.no_grad():
    for kk in range(rangege):
        image_index = 801 + kk
        #flip = numpy.random.randint(low=0, high=100)

        target_amp = tools.loadimage(path=validpath, image_index=image_index, channel=2, flip=0, m=m, n=n,
                                      convert=convert)

        real = torch.cos(init_phase * 2 * torch.pi)
        imag = torch.sin(init_phase * 2 * torch.pi)
        target_amp_complex = torch.complex(target_amp * real, target_amp * imag)

        slmfield = propagation_ASM1(u_in=target_amp_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                   wavelength=wavelength,
                                   precomped_H=Hforward)
        holo_phase = net(slmfield)

        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        recon_complex = propagation_ASM2(u_in=slm_complex, z=-z, linear_conv=pad, feature_size=[pitch, pitch],
                                        wavelength=wavelength,
                                        precomped_H=Hbackward)

        recon_amp = torch.abs(recon_complex)
        recon_amp = torch.squeeze(recon_amp)
        recon_amp = recon_amp.cpu().data.numpy()
        target_amp = torch.squeeze(target_amp)
        target_amp = target_amp.cpu().numpy()
        psnrr = psnr(recon_amp, target_amp)
        currentpsnr = currentpsnr + psnrr
        ssimm = ssim(recon_amp, target_amp)

        currentssim = currentssim + ssimm
print('avgpsnr:',currentpsnr/rangege)
print('avgssim:',currentssim/rangege)