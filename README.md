# RT-4K-CGH

## environment set up

The environment is the same as https://github.com/flyingwolfz/CCNN-CGH/tree/main

```
conda create -n ccnncgh python=3.9
conda activate ccnncgh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install opencv-python
pip install tqdm
pip install scipy
pip install scikit-image
```    
The complexPyTorch is from https://github.com/wavefrontshaping/complexPyTorch

The ASM is from https://github.com/computational-imaging/neural-holography

download the pretrained phases from BaiduNetdisk: https://pan.baidu.com/s/1yXb8KisjLYeLGKCJaljMjQ?pwd=q3fw 

or google drive: https://drive.google.com/drive/folders/1462bOb1uWxweCejxQ5OBxWYLljMMDiAu?usp=sharing
