# shanji

山脊线、山谷线提取与点云映射工具。

本项目用于从 LAS/LAZ 点云中提取地形开度图、连续山谷线和连续山脊线，并把线位置映射回点云。当前流程重点输出简单、可查看的结果，不再生成大量调试文件。

## 输入数据

- 支持 `.las` / `.laz` 点云文件。
- 点云需要包含 `x/y/z` 坐标和 `classification` 字段。
- 当前只使用 `classification = 2` 和 `classification = 16` 作为地面参考点。

`config.yaml` 中的关键配置：

```yaml
ground_classes: [2, 16]

simple_openness:
  enabled: true
  classes: [2, 16]
  output_class: 3
  point_buffer_distance: 10.0
  point_mapping_mode: "nearest_line_cell"
  points_per_line_cell: 3
```

## 输出内容

每个输入点云只输出 4 个文件：

```text
openness.tif
valley_on_openness.tif
ridge_on_openness.tif
class3_openness_features.las
```

说明：

- `openness.tif`：灰度开度图。
- `valley_on_openness.tif`：开度图上叠加山谷线位置。
- `ridge_on_openness.tif`：开度图上叠加山脊线位置。
- `class3_openness_features.las`：映射后的点云，输出点的 `classification` 全部写为 `3`。

`class3_openness_features.las` 中用 `user_data` 区分类型：

```text
1 = 山谷
2 = 山脊
3 = 山谷和山脊重叠
```

## 当前提取思路

1. 从 `classification = 2, 16` 的点生成 DTM。
2. 对 DTM 做有限距离补洞和平滑。
3. 计算正开度，输出灰度 `openness.tif`。
4. 山谷线使用 D8 流向和汇流累积追踪，输出连续曲线。
5. 山脊线使用开度峰值、剖面高点和 TPI 等指标提取连续曲线。
6. 把山谷线、山脊线栅格化后叠加到开度图。
7. 点云映射时，每个线像元在 10m 搜索范围内选取最多 3 个最近点，写入 `classification = 3`。

## 安装依赖

建议使用虚拟环境：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
laspy
numpy
scipy
rasterio
scikit-image
shapely
pyproj
pyyaml
matplotlib
Pillow
```

## 运行方式

处理 `config.yaml` 中配置的默认输入：

```bash
python main.py --config config.yaml
```

处理单个点云：

```bash
python main.py --input data/example.las --config config.yaml
```

批量处理文件夹中的点云：

```bash
python main.py --input data --config config.yaml
```

当输入是文件夹时，程序会递归查找 `.las/.laz` 文件，并跳过 `output`、`openness_output` 等输出目录。

## 输出目录规则

简单开度模式开启时，输出目录为：

```text
<输入点云所在目录>/openness_output/<点云文件名不含扩展名>/
```

例如：

```text
data/processed_01.las
data/openness_output/processed_01/
  openness.tif
  valley_on_openness.tif
  ridge_on_openness.tif
  class3_openness_features.las
```

## 常用调参

让映射点云更密：

```yaml
simple_openness:
  points_per_line_cell: 4
```

让映射点云更细、更贴线：

```yaml
simple_openness:
  points_per_line_cell: 1
  point_buffer_distance: 5.0
```

减少山谷线数量：

```yaml
valley:
  primary:
    keep_top_n: 20
  supplement:
    keep_top_n: 10

local_supplement:
  enabled: false
```

## 注意事项

- 原始点云和输出结果通常很大，已在 `.gitignore` 中排除。
- Git 仓库只建议提交代码、配置和说明文件。
- 如果查看器只能按 `classification` 着色，山脊和山谷都会显示为同一个 class3 颜色；需要区分时请查看 `user_data` 字段。
