# MC360IQA
MC360IQA: A Multi-channel CNN for Blind 360-degree Image Quality Assessment

## Usage
If you want to train the code on your database (e.g.  [CVIQ database](https://github.com/sunwei925/CVIQDatabase) ):

First, prepare the database
```sh
cd equi2cubic
ConvertCVIQtoCubic.m
```
Then
```sh
CUDA_VISIBLE_DEVICES=0 python train.py \
--num_epochs 10 \
--batch_size 40 \
--database CVIQ \
--data_dir /DATA/CVIQcubic \
--filename_train CVIQ/CVIQ_train.csv \
--filename_test CVIQ/CVIQ_test.csv \
--snapshot  /DATA/ModelFolder/VRIQA \
--cross_validation_index 1
```
If you want to test the trained model on the test set:

```sh
CUDA_VISIBLE_DEVICES=1 python test.py \
--database CVIQ \
--data_dir /DATA/CVIQcubic \
--filename_test CVIQ/CVIQ_test.csv \
--snapshot  /DATA/ModelFolder/VRIQA/CVIQ/1/CVIQ.pkl
```
If you just want to evaluate the quality of an equirectangular image:

```sh
CUDA_VISIBLE_DEVICES=0 python test_on_equirectangular.py \
--filename images/1.png \
--snapshot  /DATA/ModelFolder/VRIQA/CVIQ/1/CVIQ.pkl
```

## Model
You can download the trained model via:

CVIQ: [google drive](https://drive.google.com/file/d/13Nkw7RL9uQUWwpYnA0_rope5dSI--wgl/view?usp=sharing) [baidu yun](https://pan.baidu.com/s/18oXi5kLN0ZHwuFsVDZldlg) 提取码：5muh 

OIQA: [google drive](https://drive.google.com/file/d/1NJSAYggAwSKyP4YmZBkMlPYW5T3-mC-f/view?usp=sharing) [baidu yun](https://pan.baidu.com/s/1kwG3tp5UOP9AeiKBaqH42g 
) 提取码：39we

We recommend you to use the model trained on the OIQA database since it is more robust.

## Citation
**If you find this code is useful for  your research, please cite**:

```latex
@article{sun2019mc360iqa,
  title={MC360IQA: A Multi-channel CNN for Blind 360-degree Image Quality Assessment},
  author={Sun, Wei and Min, Xiongkuo and Zhai, Guangtao and Gu, Ke and Duan, Huiyu and Ma, Siwei},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={14},
  number={1},
  pages={64-77},
  year={2020},
  publisher={IEEE}
}
```
## Acknowledgement

1. <https://github.com/rayryeng/equi2cubic>
2. <https://github.com/pepepor123/equirectangular-to-cubemap>
3. <https://github.com/vztu/RAPIQUE>
