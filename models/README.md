# 模型（待补充）

**当前状态：空的。** 仓库里只有 `classes.txt`，没有任何权重文件。
`configs/detect.json` 里 `paths.model` 指向 `models/yolov5lite-320.onnx`，
文件不在的时候程序会自动退回占位后端（不做真实推理），整条管道照样跑得通。

## 需要放什么

| 后端 | 放的文件 | 配置里怎么写 |
| --- | --- | --- |
| onnx | `models/yolov5lite-320.onnx` | `"model": "models/yolov5lite-320.onnx"` |
| ncnn | `models/yolov5lite-320.param` + `.bin` | 仍然填 `.onnx` 结尾的名字，后端会把后缀替换掉 |

权重文件不进 Git（`.gitignore` 里排除了 `*.onnx` / `*.param` / `*.bin` / `*.kmodel`）。

## 模型输出格式要求

后处理按 YOLOv5 的常规导出格式写的，模型输出需要满足：

- 形状 `(1, N, 5 + num_classes)`，N 是候选框数量（320 输入一般是 25200 或者 6300）
- 每行依次是 `cx, cy, w, h, obj_conf, cls0, cls1, ...`
- 前四个坐标可以是归一化值（0~1），也可以是输入尺寸下的像素值，两种都兼容
- 类别分数是已经过 sigmoid 的概率，不再需要额外激活

YOLOv5 默认的 `export.py` 导出来就是这个格式。如果导出时加了 `--grid` 或者用了带 NMS 的
导出（输出直接是 `(1, N, 6)` 的最终结果），这里就解不了了，那种情况要单独改 `decode_output()`。

## 怎么导出

YOLOv5-Lite 的权重转 ONNX，在训练机上执行：

```bash
git clone https://github.com/ppogg/YOLOv5-Lite
cd YOLOv5-Lite

# 320 输入，batch 1
python export.py --weights yolov5lite-s.pt --img 320 --batch 1 --include onnx --opset 12
```

导出后确认一下输入输出名字：

```bash
python -c "import onnxruntime as ort; s=ort.InferenceSession('yolov5lite-s.onnx'); \
print([i.name for i in s.get_inputs()], [o.name for o in s.get_outputs()])"
```

名字和 `configs/detect.json` 里的 `input_name` / `output_name` 不一样也没关系，
后端会以模型里的实际名字为准，只在配置的名字不存在时打一行提示。

## 用 ncnn 的话

ncnn 在树莓派上比 onnxruntime 快，代价是要多折腾一轮：

1. 上面导出的 onnx 转 ncnn：`onnx2ncnn yolov5lite-s.onnx yolov5lite.param yolov5lite.bin`
   或者用 pnnx 重新导一遍
2. 装 ncnn 的 python 包（官方预编译轮子不一定有 aarch64 版本，多半得自己编）
3. 配置里把 `model.backend` 改成 `"ncnn"`，`input_name` / `output_name` 填 ncnn 里实际的 blob 名

没编好之前建议先用 onnx 跑起来，功能是一样的，只是帧率低一点。

## 类别文件

`classes.txt` 一行一个类名，顺序就是 `class_id`，和数据集标注里的数字对应。
这份是实验室场景的 8 类，换成自己的模型时整个替换。具体见 `data/README.md` 的类别表。

## 没有模型时的表现

- 启动日志会打印 `[model] 没找到模型文件 ...，改用占位后端（不做真实推理）`
- 画面上左下角会有一行 `no model loaded - placeholder backend`
- 统计里"检测框总数"是 0

想看看画框效果，可以加 `--fake-detections`，占位后端会返回两个固定的假框，
那个只是用来看渲染的，不要当成检测结果。
