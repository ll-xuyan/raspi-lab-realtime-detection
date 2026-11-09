# 数据集（待补充）

**当前状态：空的。** 仓库里只有目录和读取接口，没有任何图片和标注。
`data/images/` 和 `data/labels/` 下面各留了一个 `.gitkeep`，除此之外没有文件。

不管换成什么数据，只要按下面的约定放好，`src/dataset.py` 不用改一行代码。

## 放哪里

```
data/
├── images/          图片放这里
│   ├── lab_0001.jpg
│   └── lab_0002.jpg
└── labels/          标注放这里，文件名和图片一致，扩展名换成 .txt
    ├── lab_0001.txt
    └── lab_0002.txt
```

- 一一对应：`images/lab_0001.jpg` 必须配 `labels/lab_0001.txt`
- 图片格式支持 `.jpg` `.jpeg` `.png`，扩展名大小写都行
- 标注文件缺失、或者格式不对，`--check-dataset` 会报出来，不会静默跳过
- 数据量大的话不用一次全放进来，脚本是扫目录的，加文件就生效

## 标注格式

YOLO 格式，一行一个目标：

```
class_id  cx  cy  w  h
```

- `class_id` 从 0 开始，对应 `models/classes.txt` 里的行号
- `cx cy` 是框中心点，`w h` 是框宽高，四个值都除以图片宽高做了归一化，范围 0~1
- 数字之间用空格分隔，逗号也能读，但建议统一用空格
- 一张图没有目标就留一个空文件，不要删掉标注文件

举个例子，`data/labels/lab_0001.txt` 里如果有两个目标，内容长这样：

```
0 0.512500 0.421300 0.187500 0.518500
2 0.734400 0.680600 0.096900 0.148100
```

（这两行只是格式示意，不对应仓库里的任何图片。）

## 类别

`models/classes.txt` 里现在是 8 类，顺序就是 `class_id`：

| id | 类名 | 说明 |
| --- | --- | --- |
| 0 | person | 人，用来判断工位在不在岗 |
| 1 | bottle | 瓶子 |
| 2 | cup | 杯子 |
| 3 | laptop | 笔记本 |
| 4 | keyboard | 键盘 |
| 5 | mouse | 鼠标 |
| 6 | chair | 椅子 |
| 7 | cell phone | 手机 |

换自己的模型时，这个文件和数据集要一起换，顺序必须和训练时一致，否则框会标错类别。

## 标注工具

labelImg、Label Studio、Roboflow 都能直接导出 YOLO 格式，选 "YOLO" 而不是 "VOC" 或 "COCO"。
导出后一般会带一个 `classes.txt`，按它把 `models/classes.txt` 覆盖掉就行。

## 校验

```bash
python src/main.py --check-dataset
```

会打印图片数、标注数、目标总数、各类别数量和发现的问题（缺标注、行格式错误、标注全空）。
数据齐全且格式正确时最后一行是"检查通过，数据可以直接用"。

## 没数据时的行为

空目录不会让程序崩：

- `LabDataset.__len__()` 返回 0，迭代不产出任何样本
- `sample.labels` 的形状是 `(0, 5)` 的空数组，不是 `None`，下游取 `shape` 不会炸
- `python src/main.py --dataset-demo` 会用 `demo_samples()` 造 3 条占位样本，把读取接口走一遍，
  能看清一条样本的结构：图片路径、标注路径、标注数组

也就是说，现在的代码是照着"有数据"的样子写的，只是数据那格还空着。

## 说明

图片和标注都不进 Git（`.gitignore` 里排除了 `data/images/*` 和 `data/labels/*`）。
数据留在本地，仓库只保存目录约定和读取代码。
