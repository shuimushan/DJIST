# DJIST

This is the official code for the paper **"[DJIST: Decoupled joint image and sequence training framework for sequential visual place recognition](https://www.sciencedirect.com/science/article/pii/S0925231225022945)"**, which has been published in Neurocomputing.

## Dataset

For downloading and organizing the datasets, please refer to: [JIST](https://github.com/ga1i13o/JIST).

## Train

```
python train.py --dataset_folder /path/SF-XL/processed \
    --seq_dataset_path /path/msls_processed \
    --aggregation_type seqgem  \
    --img_shape 224 224 \
    --lambda_triplet 100000 \
    --lambda_im2im 100 \
    --lambda_triplet_me 100
```

### Test

```
python evaluation.py --resume_model ./weights/DJIST.pth --seq_dataset_path /path/robotcar_cut_2m_formatted --aggregation_type seqgem --img_shape 224 224
```

### Related Work

The code for SciceVPR, a Stable Cross-Image Correlation Enhanced Model for Visual Place Recognition, has been released at: [SciceVPR](https://github.com/shuimushan/SciceVPR).

### Acknowledgements

Parts of this repository are inspired by the following repositories:

*[SciceVPR](https://github.com/shuimushan/SciceVPR)\
*[JIST](https://github.com/ga1i13o/JIST)\
*[CricaVPR](https://github.com/Lu-Feng/CricaVPR)\
*[DINO-Mix](https://github.com/GaoShuang98/DINO-Mix)

## Cite
Here is the bibtex to cite our paper

```
@article{DJIST,
title = {DJIST: Decoupled joint image and sequence training framework for sequential visual place recognition},
journal = {Neurocomputing},
volume = {658},
pages = {131622},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.131622},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225022945},
author = {Shanshan Wan and Lai Kang and Yingmei Wei and Tianrui Shen and Haixuan Wang and Chao Zuo},
keywords = {Visual place recognition, Image retrieval, Multi-branch feature extraction, Attention separation loss}
}
```
