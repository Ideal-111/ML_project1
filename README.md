
# 说明

本仓库利用libsvm工具在MINIST数据集上进行训练和预测，最终在测试集上取得了98.63%的准确率✅

## Usage

```bash
python main.py
```

## 文件

- libsvm-3.36是libsvm最新版本，放在根目录下，请勿移动位置
- data中未存放MINIST数据集，请自行下载，共60000张图片（50000训练+10000测试）
- WhetherLinearSeparable.py 用于判断一个训练数据集是否线性可分
