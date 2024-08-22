# 🎹🎻🥁 Jazz Trio Database 🎹🎻🥁

![Version Badge](https://img.shields.io/badge/version-v0.2-blue) [![DOI (Paper)](http://img.shields.io/badge/DOI-10.5334/tismir.186-blue)](https://doi.org/10.5334/tismir.186) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a target="_blank" href="https://colab.research.google.com/github/HuwCheston/Cambridge-Jazz-Trio-Database/blob/main/example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The **Jazz Trio Database** is a dataset composed of about 45 hours of jazz performances annotated by an automated signal processing pipeline. For more information, [check out the docs](https://huwcheston.github.io/Jazz-Trio-Database/) or our paper published in [Transactions of the International Society for Music Information Retrieval](https://doi.org/10.5334/tismir.186).

## Contents:
- [Dataset](#dataset)
- [License](#license)
- [Citation](#citation)
- [Outputs](#outputs)

## Dataset
 
To download or build the repository from source, [read the relevant page of our docs](https://huwcheston.github.io/Jazz-Trio-Database/installation/getting-started.html). Note that audio recordings are not stored directly in this repository. Instead, when you run the script to build the dataset, these files will be downloaded automatically from an official YouTube source. Note that, occasionally, individual YouTube recordings may be taken down: these tracks will be skipped when building the dataset, but please report any issues [here](mailto:huwcheston@gmail.com?subject=CJD-Missing-YouTube-link) so that working links can be added in these cases.

## License

The dataset is made available under the [MIT License](https://spdx.org/licenses/MIT.html). Please note that your use of the audio files linked to on YouTube is not covered by the terms of this license.

## Citation

If you use the Jazz Trio Database in your work, please cite the paper where it was first introduced:

```
@article{jazz-trio-database
    title = {Jazz Trio Database: Automated Annotation of Jazz Piano Trio Recordings Processed Using Audio Source Separation},
    url = {https://doi.org/10.5334/tismir.186},
    doi = {10.5334/tismir.186},
    publisher = {Transactions of the International Society for Music Information Retrieval},
    author = {Cheston, Huw and Schlichting, Joshua L and Cross, Ian and Harrison, Peter M C},
    year = {2024},
}
```

## Outputs

The Jazz Trio Database has been used in the following published research outputs:

- Cheston, H., Schlichting, J. L., Cross, I., & Harrison, P. M. C. (2024, In Publication). Rhythmic Qualities of Jazz Improvisation Predict Performer Identity and Style in Source-Separated Audio Recordings. Royal Society Open Science.
- Cheston, H., Cross, I., & Harrison, P. M. C. (2023). An Automated Pipeline for Characterizing Timing in Jazz Trios. Proceedings of the DMRN+18 Digital Music Research Network. Digital Music Research Network, Queen Mary University of London, London, United Kingdom.
