## 机器学习


## 生成对抗网络
#### 数据集
* [数据集（提取码：g5qa）](https://pan.baidu.com/s/1eSifHcA)

#### 运行
* 训练
```
python gan_demo/main.py train --gpu=False --vis=True  --batch-size=256  --max-epoch=200
```
> vis=True 即使用 visdom，可通过 http://localhost:8097 看到生成的图像

* 使用生成网络生成图像
```
python gan_demo/main.py generate --gen-im='result1.5w.png --gen-search-num=15000
```

## 使用 Fast Neural Style 的方式实现风格迁移
> ![神经网络风格迁移](https://github.com/leonzm/py37_maching_learning/raw/master/fast_neural_style/desc/fast_neural_style.png)
