# 🎹🎻🥁 Jazz Trio Database 🎹🎻🥁

![Version Badge](https://img.shields.io/badge/version-v0.2-blue) [![DOI (Paper)](http://img.shields.io/badge/DOI-10.5334/tismir.186-blue)](https://doi.org/10.5334/tismir.186) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a target="_blank" href="https://colab.research.google.com/github/HuwCheston/Cambridge-Jazz-Trio-Database/blob/main/example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The **Jazz Trio Database** (JTD) is a dataset composed of about 45 hours of jazz performances annotated by an automated signal processing pipeline. For more information, [check out the docs](https://huwcheston.github.io/Jazz-Trio-Database/) or our paper published in [Transactions of the International Society for Music Information Retrieval](https://doi.org/10.5334/tismir.186).

## Contents:
- [Dataset](#dataset)
- [License](#license)
- [Citation](#citation)
- [Outputs](#outputs)

## Dataset
 
JTD is now integrated within the latest version of [`mirdata`](https://mirdata.readthedocs.io/en/stable/index.html), and this is the recommended way to work with the database annotations moving forwards.  To install `mirdata`, run the following code (best inside a `virtualenv`):

```
pip install git+https://github.com/mir-dataset-loaders/mirdata.git
```

Now, you can download the dataset and access the annotations simply by running the following lines of Python code:

```
import mirdata

jtd = mirdata.initialize('jtd')
jtd.download()
```

Audio recordings (both mixed and unmixed) can be found on [this Zenodo record](https://zenodo.org/records/13828030). Further instruction on where these recordings must be placed will be provided when running `jtd.download()`. Access must be requested before JTD audio can be downloaded, and will only be granted to valid research projects. Please provide as much detail relating to how you hope to use JTD when requesting access to the audio.

For more information on using `mirdata` together with JTD, refer to the [`mirdata` documentation](https://mirdata.readthedocs.io/en/stable/source/overview.html) and the examples given in [our documentation](https://huwcheston.github.io/Jazz-Trio-Database/installation/download-database.html). Although it is not recommended, to build the JTD annotations directly from source, [read the relevant page of our docs](https://huwcheston.github.io/Jazz-Trio-Database/installation/getting-started.html).

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

Further information on `mirdata` can be found in the following paper:

```
@inproceedings{
  bittner_fuentes_2019,
  title={mirdata: Software for Reproducible Usage of Datasets},
  author={Bittner, Rachel M and Fuentes, Magdalena and Rubinstein, David and Jansson, Andreas and Choi, Keunwoo and Kell, Thor},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2019}
}
```

## Outputs

The Jazz Trio Database has been used in the following published research outputs:

- Cheston, H., Bance, R., & Harrison, P. M. C. (2025). Deconstructing Jazz Piano Style Using Machine Learning. _arXiv_. https://doi.org/10.48550/arXiv.2504.05009
- Cheston, H., Schlichting, J. L., Cross, I., & Harrison, P. M. C. (2024). Rhythmic Qualities of Jazz Improvisation Predict Performer Identity and Style in Source-Separated Audio Recordings. _Royal Society Open Science_. https://doi.org/10.1098/rsos.240920
- Cheston, H., Cross, I., & Harrison, P. M. C. (2023). An Automated Pipeline for Characterizing Timing in Jazz Trios. _Proceedings of the DMRN+18 Digital Music Research Network_. Digital Music Research Network, Queen Mary University of London, London, United Kingdom.
