# 山谷线和山脊线提取程序

本项目用于从 LAS/LAZ 点云中提取山谷线和山脊线，并把结果保存为 GeoJSON、特征点 LAS 和预览图。程序会先从地面点生成 DTM，再结合汇流追踪、宽谷补充、山脊开阔度、分水岭山脊、脊线补全和后处理过滤得到最终线要素。

## 适用输入

- 输入文件格式：`.las` 或 `.laz`
- 点云必须包含 `x/y/z` 坐标
- 点云必须包含 `classification` 字段
- 默认只使用 `classification = 2` 的地面点参与分析

`config.yaml` 当前默认输入为：

```yaml
input_las: "processed_12.las"
ground_class: 2
```

也可以在运行时通过 `--input` 指定单个点云文件或包含多个点云的文件夹。

## 安装依赖

建议使用虚拟环境后安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括 `laspy`、`numpy`、`scipy`、`rasterio`、`scikit-image`、`shapely`、`pyproj`、`pyyaml`、`matplotlib` 和 `Pillow`。

## 运行方法

使用 `config.yaml` 中的默认输入：

```bash
python main.py --config config.yaml
```

指定单个点云：

```bash
python main.py --input processed_12.las --config config.yaml
```

批量处理文件夹中的全部 `.las/.laz` 点云：

```bash
python main.py --input data --config config.yaml
```

说明：

- `--config` 是可选配置覆盖文件；不传时会使用 `main.py` 内置默认配置。
- `--input` 优先级高于 `config.yaml` 里的 `input_las`。
- 当 `--input` 是文件夹时，程序会递归查找 `.las/.laz` 文件，并自动跳过路径中包含 `output` 的目录。

## 输出位置

当前 `main.py` 使用批处理输出规则，不直接使用 `config.yaml` 中的 `output_dir` 作为最终目录。每个输入点云的输出目录为：

```text
<输入点云所在目录>/output/<点云文件名不含扩展名>/
```

示例：

```text
processed_12.las
output/
  processed_12/
    terrain_features.geojson
    terrain_feature_points.las
    preview.png
```

如果输入为 `data/a.las`，则输出到：

```text
data/output/a/
```

## 输出文件

### terrain_features.geojson

最终山谷线和山脊线矢量结果，格式为 GeoJSON `FeatureCollection`。

主要属性：

| 字段 | 说明 |
| --- | --- |
| `feature_type` | 要素类型，`valley` 表示山谷线，`ridge` 表示山脊线 |
| `valley_method` | 山谷线来源，如 `flow_trace`、`broad_valley`、`major_valley` |
| `ridge_method` | 山脊线来源，当前统一写为 `ridge_openness_top_combined` |
| `importance_score` | 重要性评分，启用 `line_importance.enabled` 时写入 |
| `importance_level` | 重要性等级：`high`、`medium`、`low` |
| `extreme_ratio` | 剖面极值比例 |
| `mean_local_relief` | 平均局部起伏 |

该文件可直接在 QGIS、ArcGIS 等 GIS 软件中打开。

### terrain_feature_points.las

从原始地面点中筛选靠近最终特征线的点，并写出为新的 LAS 文件。

标记规则：

| `user_data` | 含义 |
| --- | --- |
| `1` | 山谷特征点 |
| `2` | 山脊特征点 |

默认启用重要性过滤：

```yaml
point_mapping:
  point_buffer_distance: 5.0
  use_importance_filter: true
  min_importance_level: "medium"
```

也就是说，只有达到 `medium` 或 `high` 等级的线会参与特征点映射。

### preview.png

最终预览图：

- 背景为 DTM 阴影图
- 蓝色线为山谷线
- 红色线为山脊线
- 青色虚线用于显示宽谷补充线

该图主要用于人工快速检查结果，不建议作为正式空间数据使用。

### 可选调试输出

当配置中开启相关开关时，会额外输出调试图或阶段预览图：

```yaml
output:
  save_stage_previews: false
  save_debug_images: false
  save_seed_debug: false
  save_broad_debug: false
  save_ridge_debug: false
  save_openness_debug: false
```

可能生成的文件包括：

- `preview_flow_only.png`
- `preview_broad_valley_only.png`
- `preview_ridge_watershed_divide_only.png`
- `preview_combined.png`
- `debug_dtm.png`
- `debug_support_mask.png`
- `debug_valley_accumulation.png`
- `debug_ridge_accumulation.png`
- `debug_trace_seeds.png`

## 主要流程

1. 读取 LAS/LAZ，并筛选 `ground_class` 对应的地面点。
2. 按 `dtm.resolution` 栅格化生成 DTM，每个格网取地面点高程中值。
3. 对无值区做有限距离 IDW 填补，最大距离由 `dtm.max_fill_distance` 控制。
4. 对 DTM 做 NaN 安全的高斯平滑。
5. 计算 D8 流向和汇流累积量。
6. 使用 `flow_trace_two_stage` 提取主山谷线和补充山谷线。
7. 提取宽谷线，并与流线结果做去重、裁剪和合并。
8. 通过开阔度、TPI、剖面形态、分水岭边界和谷线距离等指标提取山脊线。
9. 对山脊线执行边界过滤、断裂连接、密集线裁剪和最终剖面过滤。
10. 评估线的重要性，保存 GeoJSON、特征点 LAS 和预览图。

## 关键配置说明

### 输入和 DTM

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `input_las` | `processed_12.las` | 默认输入文件或文件夹 |
| `ground_class` | `2` | 地面点分类值 |
| `dtm.resolution` | `2.0` | DTM 分辨率，单位米 |
| `dtm.max_fill_distance` | `20.0` | DTM 无值区最大填补距离，单位米 |
| `dtm.smooth_sigma_cells` | `1.2` | 山谷分析用 DTM 平滑强度 |
| `ridge_dtm.smooth_sigma_cells` | `0.9` | 山脊分析用 DTM 平滑强度 |

### 山谷线

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `extraction.method` | `flow_trace_two_stage` | 两阶段汇流追踪 |
| `valley.primary.seed_percentile` | `97.8` | 主山谷种子阈值分位数 |
| `valley.primary.continue_percentile` | `85.0` | 主山谷延续阈值分位数 |
| `valley.primary.min_line_length` | `90.0` | 主山谷最小线长，单位米 |
| `valley.supplement.enabled` | `true` | 是否启用补充山谷线 |
| `broad_valley.enabled` | `true` | 是否启用宽谷提取 |
| `major_valley_filter.enabled` | `true` | 是否启用主山谷筛选 |

### 山脊线

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `ridge_openness_top.enabled` | `true` | 启用顶部开阔度山脊提取 |
| `broad_crest_ridge.enabled` | `true` | 启用宽缓脊补充 |
| `watershed_divide_ridge.enabled` | `true` | 启用分水岭山脊提取 |
| `ridge_center_supplement.enabled` | `true` | 启用山脊中心补充 |
| `ridge_gap_connect.enabled` | `true` | 启用山脊断裂连接 |
| `ridge_final_filter.enabled` | `true` | 启用最终剖面过滤 |
| `ridge_dense_prune.enabled` | `true` | 启用密集山脊线裁剪 |

### 边界和后处理

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `edge_filter.enabled` | `true` | 启用边界过滤 |
| `edge_filter.ridge_edge_buffer_m` | `80.0` | 山脊核心区边界缓冲距离 |
| `edge_filter.valley_edge_buffer_m` | `20.0` | 山谷边界缓冲距离 |
| `postprocess_valley.merge_distance` | `12.0` | 山谷线合并距离 |
| `postprocess_ridge.merge_distance` | `20.0` | 山脊线合并距离 |
| `postprocess_ridge.max_merge_angle_deg` | `50.0` | 山脊线最大合并角度 |

## 常用调参建议

减少山谷线数量：

```yaml
valley:
  primary:
    seed_percentile: 98.5
    continue_percentile: 88.0
```

增加山谷线数量：

```yaml
valley:
  primary:
    seed_percentile: 96.5
    continue_percentile: 80.0
  supplement:
    enabled: true
```

减少边缘附近的误提取山脊：

```yaml
edge_filter:
  ridge_edge_buffer_m: 100.0
  endpoint_edge_buffer_m: 90.0
```

让特征点 LAS 标记更多点：

```yaml
point_mapping:
  point_buffer_distance: 8.0
  min_importance_level: "low"
```

提高 DTM 精细度但增加运行时间：

```yaml
dtm:
  resolution: 1.0
```

降低运行时间但结果更粗：

```yaml
dtm:
  resolution: 5.0
```

## 常见问题

### 没有生成 `terrain_feature_points.las`

检查：

- `output.save_feature_points` 是否为 `true`
- `line_importance.enabled` 和 `point_mapping.min_importance_level` 是否过滤过严
- `point_mapping.point_buffer_distance` 是否过小
- 输入点云是否有足够的 `classification = 2` 地面点

### 输出线太靠近数据边缘

增大边界过滤距离：

```yaml
edge_filter:
  ridge_edge_buffer_m: 100.0
  valley_edge_buffer_m: 30.0
  endpoint_edge_buffer_m: 90.0
```

### 批量处理时输出在哪里

每个输入文件都会在自己的同级 `output/<文件名>/` 目录下生成结果。例如：

```text
data/tile_001.las -> data/output/tile_001/
data/tile_002.laz -> data/output/tile_002/
```

### 修改 `config.yaml` 后是否需要重新运行

需要。修改参数后重新执行：

```bash
python main.py --config config.yaml
```

## 项目文件

```text
D:\xishudimiandian\shanjixiantiqu\
  main.py            主程序
  config.yaml        配置文件
  requirements.txt   Python 依赖
  README.md          使用说明
  processed_12.las   示例/默认输入点云
  data\              可选批处理输入目录
```

## 版本

- 更新时间：2026-06-02
- 当前说明基于 `main.py` 和 `config.yaml` 更新

## Dual-source structure fusion

`dual_source.enabled: true` enables the new two-source workflow in the same LAS/LAZ file:

- `candidate_source`: uses classes `[2, 1]`, excludes `[4, 16]`, and only uses class `1` in grid cells that do not contain class `2`.
- `pass1_source`: uses classes `[2, 16]`, excludes `[1, 4]`.
- Both sources share one grid/transform/shape from `grid.bounds_source`.
- The legacy single-source workflow remains available by setting `dual_source.enabled: false`.

Run:

```bash
python main.py --input processed_12.las --config config.yaml
```

Main dual-source outputs are written under each tile output folder:

```text
output/<tile>/
  candidate_source/
    candidate_terrain_features.geojson
    candidate_ridge_lines.geojson
    candidate_valley_lines.geojson
    candidate_preview.png
    candidate_class_usage.json
    candidate_class1_used_mask.tif
  pass1_source/
    pass1_terrain_features.geojson
    pass1_ridge_lines.geojson
    pass1_valley_lines.geojson
    pass1_preview.png
  fused/
    fused_ridge_zone.tif
    fused_valley_zone.tif
    fused_structure_zone.tif
    high_confidence_ridge_zone.tif
    high_confidence_valley_zone.tif
    high_confidence_structure_zone.tif
    structure_source.tif
    structure_confidence.tif
    structure_conflict_mask.tif
    ridge_distance.tif
    valley_distance.tif
    fused_structure_preview.png
    structure_summary.json
    fused_structure_points.las
    fused_ridge_points.las
    fused_valley_points.las
    fused_conflict_points.las
    structure_point_summary.json
```

The fused point LAS files preserve the original `classification` values and use `user_data` for review labels:

- `1`: valley
- `2`: ridge
- `3`: ridge and valley overlap
- `4`: conflict
