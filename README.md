## Introduction
Open-CD is an open source change detection toolbox based on a series of open source general vision task tools.
Forked repo https://github.com/likyoo/open-cd/

#### simple usage

```
git clone git@github.com:AgustinNormand/open-cd.git
cd open-cd
git checkout dev-1.x-3PNG-1PNG
```

```
# Install OpenMMLab Toolkits as Python packages
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain>=1.0.0rc6"
pip install "mmsegmentation>=1.0.0rc6, <1.1.0."
pip install "mmdet>=3.0.0rc6, <3.1.0"

pip install -v -e .
```

train
```
python tools/train.py configs/tinycd/tinycd_256x256_40k_chaco.py --work-dir ./tinycd_chaco_workdir
```
infer
```
# get .png results
python tools/test.py configs/tinycd/tinycd_256x256_40k_chaco.py  tinycd_chaco_workdir/latest.pth --show-dir tmp_infer
# get metrics
python tools/test.py configs/tinycd/tinycd_256x256_40k_chaco.py  tinycd_chaco_workdir/latest.pth
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{fang2022changer,
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  author={Sheng Fang and Kaiyu Li and Zhe Li},
  year={2022},
  eprint={2209.08290},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

Open-CD is released under the Apache 2.0 license.
