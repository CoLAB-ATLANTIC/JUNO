# JUNO
<div id="top"></div>

## Front Detection Algorithms
<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Website][website-shield]][website-url]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

<!-- LICENSE: replace with your license url -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/CoLAB-ATLANTIC/Template/blob/master/LICENSE.txt

<!-- LINKEDIN -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/colabatlantic/

<!-- Website: Replace with projects website (if any) or leave +ATL website -->
[website-shield]: https://img.shields.io/badge/-Website-black.svg?style=for-the-badge
[website-url]: https://colabatlantic.com/

**Summary:** This project, made in partnership with +ATLANTIC CoLAB associate FEUP is a part of the JUNO project and it aims to provide different algorithms implemented in Python for the detection of oceanic fronts and the calculation of Frontal Probabilities on SST (Sea Surface Temperature) images at a given geographical location for a given period. This repository was supported by "JUNO - Robotic exploration of Atlantic waters" project from the *Fundação Luso-Americana para o Desenvolvimento* ([FLAD](https://www.flad.pt/en/))

![Image of Project](/images/two_logos.png)

<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->

## About the Project

The landscape of scientific and technological endeavours will change significantly because of the amplification power of networking. Geographic distance is no longer a significant barrier; open-source and open science are "force multipliers". Inspired by open-source principles, the +ATLANTIC team worked on the search, revision and implementation of the main algorithms developed to detect and follow oceanic fronts in the last months. Some were hidden behind several years of dust, and others are uncovered in the open-source world. The search process also included in-person and remote meetings with several researchers who collaborated gently with us to put our idea into practice. After concluding the search, the three main and most used algorithms to detect thermal fronts in the ocean were aggregated, simplified, and adapted partly to a single programming language, Python. The three algorithms chosen were: Canny, BOA and Cayula-Cornillon. They were all applied to historical data (frontal probability maps) and near-real-time both from models and satellite sea surface temperature products. 


### Canny Algorithm

It is the most widely used gradient-based algorithm for edge detection in 2D images, developed in 1986. Canny comprises several phases: noise reduction and finding the intensity gradient of the image; non-maximum suppression (which converts thick edges into thin ones); and hysteresis thresholding (to decide which edges are edges and which are not) ([LINK](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)).

In our case, for the detection of fronts in an image, the OpenCV (computer vision library) function .canny() is used based on the Ren et al. (2021) study. Several parameters were optimized like the 8-bit image, the minimum and maximum threshold for hysteresis, the aperture size of the Sobel operator (3x3, 5x5 or 7x7), and the L2gradient (a boolean for calculating the magnitude gradient of the image). For each image (matrix with SST values), a Gaussian filter might be applied, and a mask for the definition of the continental zone. With the application of Canny, the algorithm will return an array of pixel values. If a given pixel has been identified as a front, its value will be 1. Otherwise, it will be 0. Depending on the characteristics of the data we are using, namely the resolution, it will be necessary to vary some parameters of the canny_visualization function to obtain the best possible visualization. That trimming was significant in the case of Multi-Scale Ultra High Resolution (MUR) Sea Surface Temperature (SST) data. We applied a gaussian filter with a specific sigma value to reduce image noise, adapted the aperture size to a 5x5 array instead of 3x3, and changed the threshold limits. It also might be helpful to modify parameters that define the data range that the colourmap covers

We want to acknowledge the collaboration of Dr Shihe Ren, who gently shared his [code based on Matlab scripts](https://github.com/cdmpbp123/frontal_detection) which we adapted to Python.

[Canny frontal probabilities notebook](notebooks/canny_frontal_prob.ipynb)

### Belkin-O'Reilly Algorithm (BOA)

Like Canny, it is also a gradient-based algorithm whose main novelty is using a median filter that simultaneously eliminates noise and preserves fronts. It was developed by Belkin and O'Reilly (2009) and is used for chlorophyll and SST images.  

The code for the BOA [implementation was developed in R](https://rdrr.io/github/galuardi/boaR/man/boa.html) by Benjamin Galuardi (NOAA), and it was then necessary to convert it to Python. In the code, the most critical operations are: finding peaks in 5 directions and the maximum of a 5x5 sliding window, performing a median smoothed on the grid, and applying a mask to define the continental zone.  
  
Simplistically, applying the BOA algorithm to SST images allows us to obtain a matrix in which the value of the pixels corresponds to half the temperature difference between adjacent pixels. Thus, if the SST value of a pixel is 20 and the next pixel has a value of 18, then the difference is 2, so the algorithm will return 1 for this pixel.
 
We have to define a threshold value to obtain a frontal probability matrix. If the resulting pixel value is greater than the threshold, it's considered a front; otherwise, it's not.

[BOA frontal probabilities notebook](notebooks/BOA_frontal_prob.ipynb)

### Cayula-Cornillon Algorithm (CCA)

The Cayula-Cornillon algorithm (CCA) is satellite oceanography's most sophisticated edge detection algorithm (Belkin, 2021). It was developed by Jean-François Cayula and Peter Cornillon in the early 90s (Cayula and Cornillon, 1992, 1995). CCA is based on a histogram approach: a histogram of SST values of all pixels within an image of a front separating two water masses, M1 and M2, would always have two modes corresponding to the water masses M1 and M2, while the front is a place of SST values that correspond to a minimum between the two modes. 

The algorithm presented is used as a Single Image Edge Detector (SIED). Its basic idea is to use overlapping windows to investigate the statistical likelihood of an edge. That trend is found by calculating the histogram and detecting its bimodality and edge cohesiveness. The CCA relies on a combination of methods and operates at the picture, the window and the local level. The resulting edge is not based on the absolute strength of the front but on the relative strength depending on the context, thus making the edge detection temperature scale invariant.

CCA is probably the most used algorithm in studies about ocean fronts, even in the Portuguese oceanic region (Relvas et al., 2007). The programming implementation was based on Fortran or Matlab routines which are old or have no open-source usage. But both were used as the source of information for our execution in Python. We want to acknowledge Prof. Joaquim Luís (University of Algarve) for making the CCA [code available in Matlab](https://github.com/joa-quim/mirone/blob/master/src_figs/cayula_cornillon.m) through [Mirone software](http://joa-quim.pt/mirone/main.html) and Dr Paulo Oliveira (IPMA) for the Fortran-based code and guidance on the Python implementation. 

[CCA frontal probabilities notebook](notebooks/CayulaCornillon_frontal_prob.ipynb)

### References

Belkin, I. M., & O'Reilly, J. E. (2009). An algorithm for oceanic front detection in chlorophyll and SST satellite imagery. *Journal of Marine Systems, 78(3)*, 319-326.

Belkin, I. M. (2021). Remote sensing of ocean fronts in marine ecology and fisheries. *Remote Sensing*, 13(5), 883.

Cayula, J. F., & Cornillon, P. (1992). Edge detection algorithm for SST images. *Journal of Atmospheric and Oceanic technology*, 9(1), 67-80.

Cayula, J. F., & Cornillon, P. (1995). Multi-image edge detection for SST images. *Journal of Atmospheric and Oceanic Technology*, 12(4), 821-829. 

Ren, S., Zhu, X., Drevillon, M., Wang, H., Zhang, Y., Zu, Z., & Li, A. (2021). Detection of SST fronts from a high-resolution model and its preliminary results in the south China sea. *Journal of Atmospheric and Oceanic Technology*, 38(2), 387-403.

Relvas, P., Barton, E. D., Dubert, J., Oliveira, P. B., Peliz, A., Da Silva, J. C. B., & Santos, A. M. P. (2007). Physical oceanography of the western Iberia ecosystem: latest views and challenges. *Progress in Oceanography*, 74(2-3), 149-173.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Luís Figueiredo - luis.figueiredo@colabatlantic.com

Nuno Loureiro - nuno.loureiro@colabatlantic.com

Caio Fonteles - caio.fonteles@colabatlantic.com

Renato Mendes - renato.mendes@colabatlantic.com

<p align="right">(<a href="#top">back to top</a>)</p>
