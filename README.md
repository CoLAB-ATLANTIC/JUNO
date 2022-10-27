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

The goal of this repository is to provide different algorithms that allow the detection of fronts and the respective calculation of front probabilities for a given period at a given location.

3 algorithms are available in Python: Canny, BOA and Cayula-Cornillon whose implementation will allow the user to create a catalog/map of Frontal Probabilities for a specific period in a certain geographical location, through SST images.

### Canny Algorithm

Canny is the most widely used gradient-based algorithm for edge detection in 2D images, having been developed in 1986.Canny comprises several phases: noise reduction, finding the intensity gradient of the image, non-maximum suppression (which converts thick edges into thin ones) and hysteresis thresholding (to decide which edges are really edges and which are not).
For this project, the OpenCV function .canny() is used, which has as parameters the 8-bit image, the minimum and maximum Threshold for hysteresis, the aperture size of the sobel operator (3x3, 5x5 or 7x7), and the L2gradient (a boolean for calculating the magnitude gradient of the image). For each image (matrix with SST values) a Gaussian filter is applied (because the openCV canny does not apply this filter) and a mask for the definition of the continental zone.

[Canny frontal probabilities notebook](notebooks/canny_frontal_prob.ipynb)

### Belkin-O'Reilly Algorithm (BOA)

BOA is also a gradient-based algorithm, whose main novelty is the use of a median filter that simultaneously eliminates noise and preserves fronts. It was developed by Belkin and O'Reilly in 2009 and is used for chlorophyll and SST images. The code for the BOA implementation was developed in R by Galuardi.

[BOA frontal probabilities notebook](notebooks/BOA_frontal_prob.ipynb)

### Cayula-Cornillon Algorithm (CCA)

Unlike the 2 previous algorithms the Cayula-Cornillon Algorithm (CCA) uses an histogram approach, being considered the most sophisticated edge detection algorithm in satellite oceanography. The algorithm can be used as a Single Image Edge Detector (SIED) or be applied to several images in order to calculate frontal probability. Its basic idea is to use overlapping windows to investigate the statistical likelihood of an edge by 1) calculating the histogram and detect bimodality of the histogram and 2) detecting the cohesiveness of the potential edge. The CCA relies on a combination of methods and it operates at the picture, the window and the local level. The resulting edge is not based on the absolute strenght of the front, but on the relative strenght depending on the context, thus making edge detection temperature scale invariant.

[CCA frontal probabilities notebook](notebooks/CayulaCornillon_frontal_prob.ipynb)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Luís Figueiredo - Luis.figueiredo@colabatlantic.com

Nuno Loureiro - Nuno.loureiro@colabatlantic.com

Renato Mendes - renato.mendes@colabatlantic.com

<p align="right">(<a href="#top">back to top</a>)</p>
