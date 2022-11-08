# Removal of Gaussian DistributedNoise in Images with Deep Neural Network Models

The paper **Removal of Gaussian DistributedNoise in Images with Deep Neural Network Models** accepted to the 2022 30th Signal Processing and Communications Applications Conference (SIU)


To access the paper:
* [Paper](https://ieeexplore.ieee.org/document/9864913) 

To cite the paper or code:
```bibtex
@INPROCEEDINGS{9864913,
  author={Kuş, Zeki and Aydin, Musa},
  booktitle={2022 30th Signal Processing and Communications Applications Conference (SIU)}, 
  title={Removal of Gaussian Distributed Noise in Images with Deep Neural Network Models}, 
  year={2022},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SIU55565.2022.9864913}}
```

To contact authors for queries reqarding the paper:
* Zeki Kuş (zkus@fsm.edu.tr)
* Musa Aydın (maydin@fsm.edu.tr)

The removal of noise caused by environmental factors in microscopic imaging studies has become an important research topic in the field of medical imaging. In the medical imaging stage made with any digital microscopy method (Confocal, Fluorescence, etc.), undesirable noises are added to the image obtained due to factors stemming from excessive or low illumination, high or low temperature, or electronic circuit equipment. The most basic noise model formed due to these environmental factors mentioned is the Gaussian normal distribution or a characteristic function close to this distribution. It is widely known that spatial filters (mean, median, Gaussian smoothing) are applied to eliminate Gaussian noise in digital image processing. However, undesirable results may occur in the images obtained when spatial filters are used to remove the noise in the images. In particular, because high frequencies are suppressed in images where spatial filters are applied, details are lost in the final image, and a blurred image is obtained. For this reason, four different convolutional neural network-based models are used for noise removal and to improve the PSNR values in this study. As a result, the modified U-Net improved the PSNR values for different noise levels as follows: +6.23, +7.88 and +10.52 dB
