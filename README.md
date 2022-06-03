# Colorizing & Cartoonizing-GAN

## Generated Output - Implementation
![image](https://user-images.githubusercontent.com/62787572/171770663-71c51684-fc84-41d7-acf5-3c6c36f55b2b.png)

# Software installation

Clone this repository:
```bash
git clone https://github.com/jaepoong/Colorization-GAN.git
cd Colorization-GAN
```
Install dependency:
```bash
conda create -n Colorization-GAN python=3.6.7
conda activate Colorization-GAN
pip install requirments.txt
```


# Datasets 
## AFHQ
I provide a script to download afhq datasets. The datasets will be downloaded in the `data` directories.\
In this code, i used AFHQ for Colorizing

<b>AFHQ.</b> To download the [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) dataset run the following commands:
```bash
bash download.sh afhq-dataset
```
<p align="left"><img width="70%" src="assets/afhq.png" /></p>

## Landsacpe
In this code, i used Landscape for original image of cartoonizing\
<b>Landscape.</b> To download the Landscape Dataset please click below link

<b>[Kaggle Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)<b>
<p align="left"><img width="70%" src="assets/landscape.png" /></p>
  
## Cartoon Dataset
For cartooning target Dataset, i made caroon dataset by below code\
If you want make custom dataset, please implement below code after download video.
This code split the video by frame.
```bash
python video_save.py
```
By Copyright, i cant publish Cartoon dataset. If you dont have, i recommend use upright code.
In implementation, i used the 4500 Shinkai Makoto movie frame dataset. 
  
*example*
<p align="left"><img width="70%" src="assets/anime.png" /></p>
  
