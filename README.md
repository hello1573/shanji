# shanji

# 灞辫胺绾垮拰灞辫剨绾挎彁鍙栫▼搴?

鏈」鐩敤浜庝粠 LAS/LAZ 鐐逛簯涓彁鍙栧北璋风嚎鍜屽北鑴婄嚎锛屽苟鎶婄粨鏋滀繚瀛樹负 GeoJSON銆佺壒寰佺偣 LAS 鍜岄瑙堝浘銆傜▼搴忎細鍏堜粠鍦伴潰鐐圭敓鎴?DTM锛屽啀缁撳悎姹囨祦杩借釜銆佸璋疯ˉ鍏呫€佸北鑴婂紑闃斿害銆佸垎姘村箔灞辫剨銆佽剨绾胯ˉ鍏ㄥ拰鍚庡鐞嗚繃婊ゅ緱鍒版渶缁堢嚎瑕佺礌銆?

## 閫傜敤杈撳叆

- 杈撳叆鏂囦欢鏍煎紡锛歚.las` 鎴?`.laz`
- 鐐逛簯蹇呴』鍖呭惈 `x/y/z` 鍧愭爣
- 鐐逛簯蹇呴』鍖呭惈 `classification` 瀛楁
- 榛樿鍙娇鐢?`classification = 2` 鐨勫湴闈㈢偣鍙備笌鍒嗘瀽

`config.yaml` 褰撳墠榛樿杈撳叆涓猴細

```yaml
input_las: "processed_12.las"
ground_class: 2
```

涔熷彲浠ュ湪杩愯鏃堕€氳繃 `--input` 鎸囧畾鍗曚釜鐐逛簯鏂囦欢鎴栧寘鍚涓偣浜戠殑鏂囦欢澶广€?

## 瀹夎渚濊禆

寤鸿浣跨敤铏氭嫙鐜鍚庡畨瑁呬緷璧栵細

```bash
pip install -r requirements.txt
```

涓昏渚濊禆鍖呮嫭 `laspy`銆乣numpy`銆乣scipy`銆乣rasterio`銆乣scikit-image`銆乣shapely`銆乣pyproj`銆乣pyyaml`銆乣matplotlib` 鍜?`Pillow`銆?

## 杩愯鏂规硶

浣跨敤 `config.yaml` 涓殑榛樿杈撳叆锛?

```bash
python main.py --config config.yaml
```

鎸囧畾鍗曚釜鐐逛簯锛?

```bash
python main.py --input processed_12.las --config config.yaml
```

鎵归噺澶勭悊鏂囦欢澶逛腑鐨勫叏閮?`.las/.laz` 鐐逛簯锛?

```bash
python main.py --input data --config config.yaml
```

璇存槑锛?

- `--config` 鏄彲閫夐厤缃鐩栨枃浠讹紱涓嶄紶鏃朵細浣跨敤 `main.py` 鍐呯疆榛樿閰嶇疆銆?
- `--input` 浼樺厛绾ч珮浜?`config.yaml` 閲岀殑 `input_las`銆?
- 褰?`--input` 鏄枃浠跺す鏃讹紝绋嬪簭浼氶€掑綊鏌ユ壘 `.las/.laz` 鏂囦欢锛屽苟鑷姩璺宠繃璺緞涓寘鍚?`output` 鐨勭洰褰曘€?

## 杈撳嚭浣嶇疆

褰撳墠 `main.py` 浣跨敤鎵瑰鐞嗚緭鍑鸿鍒欙紝涓嶇洿鎺ヤ娇鐢?`config.yaml` 涓殑 `output_dir` 浣滀负鏈€缁堢洰褰曘€傛瘡涓緭鍏ョ偣浜戠殑杈撳嚭鐩綍涓猴細

```text
<杈撳叆鐐逛簯鎵€鍦ㄧ洰褰?/output/<鐐逛簯鏂囦欢鍚嶄笉鍚墿灞曞悕>/
```

绀轰緥锛?

```text
processed_12.las
output/
  processed_12/
    terrain_features.geojson
    terrain_feature_points.las
    preview.png
```

濡傛灉杈撳叆涓?`data/a.las`锛屽垯杈撳嚭鍒帮細

```text
data/output/a/
```

## 杈撳嚭鏂囦欢

### terrain_features.geojson

鏈€缁堝北璋风嚎鍜屽北鑴婄嚎鐭㈤噺缁撴灉锛屾牸寮忎负 GeoJSON `FeatureCollection`銆?

涓昏灞炴€э細

| 瀛楁 | 璇存槑 |
| --- | --- |
| `feature_type` | 瑕佺礌绫诲瀷锛宍valley` 琛ㄧず灞辫胺绾匡紝`ridge` 琛ㄧず灞辫剨绾?|
| `valley_method` | 灞辫胺绾挎潵婧愶紝濡?`flow_trace`銆乣broad_valley`銆乣major_valley` |
| `ridge_method` | 灞辫剨绾挎潵婧愶紝褰撳墠缁熶竴鍐欎负 `ridge_openness_top_combined` |
| `importance_score` | 閲嶈鎬ц瘎鍒嗭紝鍚敤 `line_importance.enabled` 鏃跺啓鍏?|
| `importance_level` | 閲嶈鎬х瓑绾э細`high`銆乣medium`銆乣low` |
| `extreme_ratio` | 鍓栭潰鏋佸€兼瘮渚?|
| `mean_local_relief` | 骞冲潎灞€閮ㄨ捣浼?|

璇ユ枃浠跺彲鐩存帴鍦?QGIS銆丄rcGIS 绛?GIS 杞欢涓墦寮€銆?

### terrain_feature_points.las

浠庡師濮嬪湴闈㈢偣涓瓫閫夐潬杩戞渶缁堢壒寰佺嚎鐨勭偣锛屽苟鍐欏嚭涓烘柊鐨?LAS 鏂囦欢銆?

鏍囪瑙勫垯锛?

| `user_data` | 鍚箟 |
| --- | --- |
| `1` | 灞辫胺鐗瑰緛鐐?|
| `2` | 灞辫剨鐗瑰緛鐐?|

榛樿鍚敤閲嶈鎬ц繃婊わ細

```yaml
point_mapping:
  point_buffer_distance: 5.0
  use_importance_filter: true
  min_importance_level: "medium"
```

涔熷氨鏄锛屽彧鏈夎揪鍒?`medium` 鎴?`high` 绛夌骇鐨勭嚎浼氬弬涓庣壒寰佺偣鏄犲皠銆?

### preview.png

鏈€缁堥瑙堝浘锛?

- 鑳屾櫙涓?DTM 闃村奖鍥?
- 钃濊壊绾夸负灞辫胺绾?
- 绾㈣壊绾夸负灞辫剨绾?
- 闈掕壊铏氱嚎鐢ㄤ簬鏄剧ず瀹借胺琛ュ厖绾?

璇ュ浘涓昏鐢ㄤ簬浜哄伐蹇€熸鏌ョ粨鏋滐紝涓嶅缓璁綔涓烘寮忕┖闂存暟鎹娇鐢ㄣ€?

### 鍙€夎皟璇曡緭鍑?

褰撻厤缃腑寮€鍚浉鍏冲紑鍏虫椂锛屼細棰濆杈撳嚭璋冭瘯鍥炬垨闃舵棰勮鍥撅細

```yaml
output:
  save_stage_previews: false
  save_debug_images: false
  save_seed_debug: false
  save_broad_debug: false
  save_ridge_debug: false
  save_openness_debug: false
```

鍙兘鐢熸垚鐨勬枃浠跺寘鎷細

- `preview_flow_only.png`
- `preview_broad_valley_only.png`
- `preview_ridge_watershed_divide_only.png`
- `preview_combined.png`
- `debug_dtm.png`
- `debug_support_mask.png`
- `debug_valley_accumulation.png`
- `debug_ridge_accumulation.png`
- `debug_trace_seeds.png`

## 涓昏娴佺▼

1. 璇诲彇 LAS/LAZ锛屽苟绛涢€?`ground_class` 瀵瑰簲鐨勫湴闈㈢偣銆?
2. 鎸?`dtm.resolution` 鏍呮牸鍖栫敓鎴?DTM锛屾瘡涓牸缃戝彇鍦伴潰鐐归珮绋嬩腑鍊笺€?
3. 瀵规棤鍊煎尯鍋氭湁闄愯窛绂?IDW 濉ˉ锛屾渶澶ц窛绂荤敱 `dtm.max_fill_distance` 鎺у埗銆?
4. 瀵?DTM 鍋?NaN 瀹夊叏鐨勯珮鏂钩婊戙€?
5. 璁＄畻 D8 娴佸悜鍜屾眹娴佺疮绉噺銆?
6. 浣跨敤 `flow_trace_two_stage` 鎻愬彇涓诲北璋风嚎鍜岃ˉ鍏呭北璋风嚎銆?
7. 鎻愬彇瀹借胺绾匡紝骞朵笌娴佺嚎缁撴灉鍋氬幓閲嶃€佽鍓拰鍚堝苟銆?
8. 閫氳繃寮€闃斿害銆乀PI銆佸墫闈㈠舰鎬併€佸垎姘村箔杈圭晫鍜岃胺绾胯窛绂荤瓑鎸囨爣鎻愬彇灞辫剨绾裤€?
9. 瀵瑰北鑴婄嚎鎵ц杈圭晫杩囨护銆佹柇瑁傝繛鎺ャ€佸瘑闆嗙嚎瑁佸壀鍜屾渶缁堝墫闈㈣繃婊ゃ€?
10. 璇勪及绾跨殑閲嶈鎬э紝淇濆瓨 GeoJSON銆佺壒寰佺偣 LAS 鍜岄瑙堝浘銆?

## 鍏抽敭閰嶇疆璇存槑

### 杈撳叆鍜?DTM

| 鍙傛暟 | 褰撳墠鍊?| 璇存槑 |
| --- | --- | --- |
| `input_las` | `processed_12.las` | 榛樿杈撳叆鏂囦欢鎴栨枃浠跺す |
| `ground_class` | `2` | 鍦伴潰鐐瑰垎绫诲€?|
| `dtm.resolution` | `2.0` | DTM 鍒嗚鲸鐜囷紝鍗曚綅绫?|
| `dtm.max_fill_distance` | `20.0` | DTM 鏃犲€煎尯鏈€澶у～琛ヨ窛绂伙紝鍗曚綅绫?|
| `dtm.smooth_sigma_cells` | `1.2` | 灞辫胺鍒嗘瀽鐢?DTM 骞虫粦寮哄害 |
| `ridge_dtm.smooth_sigma_cells` | `0.9` | 灞辫剨鍒嗘瀽鐢?DTM 骞虫粦寮哄害 |

### 灞辫胺绾?

| 鍙傛暟 | 褰撳墠鍊?| 璇存槑 |
| --- | --- | --- |
| `extraction.method` | `flow_trace_two_stage` | 涓ら樁娈垫眹娴佽拷韪?|
| `valley.primary.seed_percentile` | `97.8` | 涓诲北璋风瀛愰槇鍊煎垎浣嶆暟 |
| `valley.primary.continue_percentile` | `85.0` | 涓诲北璋峰欢缁槇鍊煎垎浣嶆暟 |
| `valley.primary.min_line_length` | `90.0` | 涓诲北璋锋渶灏忕嚎闀匡紝鍗曚綅绫?|
| `valley.supplement.enabled` | `true` | 鏄惁鍚敤琛ュ厖灞辫胺绾?|
| `broad_valley.enabled` | `true` | 鏄惁鍚敤瀹借胺鎻愬彇 |
| `major_valley_filter.enabled` | `true` | 鏄惁鍚敤涓诲北璋风瓫閫?|

### 灞辫剨绾?

| 鍙傛暟 | 褰撳墠鍊?| 璇存槑 |
| --- | --- | --- |
| `ridge_openness_top.enabled` | `true` | 鍚敤椤堕儴寮€闃斿害灞辫剨鎻愬彇 |
| `broad_crest_ridge.enabled` | `true` | 鍚敤瀹界紦鑴婅ˉ鍏?|
| `watershed_divide_ridge.enabled` | `true` | 鍚敤鍒嗘按宀北鑴婃彁鍙?|
| `ridge_center_supplement.enabled` | `true` | 鍚敤灞辫剨涓績琛ュ厖 |
| `ridge_gap_connect.enabled` | `true` | 鍚敤灞辫剨鏂杩炴帴 |
| `ridge_final_filter.enabled` | `true` | 鍚敤鏈€缁堝墫闈㈣繃婊?|
| `ridge_dense_prune.enabled` | `true` | 鍚敤瀵嗛泦灞辫剨绾胯鍓?|

### 杈圭晫鍜屽悗澶勭悊

| 鍙傛暟 | 褰撳墠鍊?| 璇存槑 |
| --- | --- | --- |
| `edge_filter.enabled` | `true` | 鍚敤杈圭晫杩囨护 |
| `edge_filter.ridge_edge_buffer_m` | `80.0` | 灞辫剨鏍稿績鍖鸿竟鐣岀紦鍐茶窛绂?|
| `edge_filter.valley_edge_buffer_m` | `20.0` | 灞辫胺杈圭晫缂撳啿璺濈 |
| `postprocess_valley.merge_distance` | `12.0` | 灞辫胺绾垮悎骞惰窛绂?|
| `postprocess_ridge.merge_distance` | `20.0` | 灞辫剨绾垮悎骞惰窛绂?|
| `postprocess_ridge.max_merge_angle_deg` | `50.0` | 灞辫剨绾挎渶澶у悎骞惰搴?|

## 甯哥敤璋冨弬寤鸿

鍑忓皯灞辫胺绾挎暟閲忥細

```yaml
valley:
  primary:
    seed_percentile: 98.5
    continue_percentile: 88.0
```

澧炲姞灞辫胺绾挎暟閲忥細

```yaml
valley:
  primary:
    seed_percentile: 96.5
    continue_percentile: 80.0
  supplement:
    enabled: true
```

鍑忓皯杈圭紭闄勮繎鐨勮鎻愬彇灞辫剨锛?

```yaml
edge_filter:
  ridge_edge_buffer_m: 100.0
  endpoint_edge_buffer_m: 90.0
```

璁╃壒寰佺偣 LAS 鏍囪鏇村鐐癸細

```yaml
point_mapping:
  point_buffer_distance: 8.0
  min_importance_level: "low"
```

鎻愰珮 DTM 绮剧粏搴︿絾澧炲姞杩愯鏃堕棿锛?

```yaml
dtm:
  resolution: 1.0
```

闄嶄綆杩愯鏃堕棿浣嗙粨鏋滄洿绮楋細

```yaml
dtm:
  resolution: 5.0
```

## 甯歌闂

### 娌℃湁鐢熸垚 `terrain_feature_points.las`

妫€鏌ワ細

- `output.save_feature_points` 鏄惁涓?`true`
- `line_importance.enabled` 鍜?`point_mapping.min_importance_level` 鏄惁杩囨护杩囦弗
- `point_mapping.point_buffer_distance` 鏄惁杩囧皬
- 杈撳叆鐐逛簯鏄惁鏈夎冻澶熺殑 `classification = 2` 鍦伴潰鐐?

### 杈撳嚭绾垮お闈犺繎鏁版嵁杈圭紭

澧炲ぇ杈圭晫杩囨护璺濈锛?

```yaml
edge_filter:
  ridge_edge_buffer_m: 100.0
  valley_edge_buffer_m: 30.0
  endpoint_edge_buffer_m: 90.0
```

### 鎵归噺澶勭悊鏃惰緭鍑哄湪鍝噷

姣忎釜杈撳叆鏂囦欢閮戒細鍦ㄨ嚜宸辩殑鍚岀骇 `output/<鏂囦欢鍚?/` 鐩綍涓嬬敓鎴愮粨鏋溿€備緥濡傦細

```text
data/tile_001.las -> data/output/tile_001/
data/tile_002.laz -> data/output/tile_002/
```

### 淇敼 `config.yaml` 鍚庢槸鍚﹂渶瑕侀噸鏂拌繍琛?

闇€瑕併€備慨鏀瑰弬鏁板悗閲嶆柊鎵ц锛?

```bash
python main.py --config config.yaml
```

## 椤圭洰鏂囦欢

```text
D:\xishudimiandian\shanjixiantiqu\
  main.py            涓荤▼搴?
  config.yaml        閰嶇疆鏂囦欢
  requirements.txt   Python 渚濊禆
  README.md          浣跨敤璇存槑
  processed_12.las   绀轰緥/榛樿杈撳叆鐐逛簯
  data\              鍙€夋壒澶勭悊杈撳叆鐩綍
```

## 鐗堟湰

- 鏇存柊鏃堕棿锛?026-06-02
- 褰撳墠璇存槑鍩轰簬 `main.py` 鍜?`config.yaml` 鏇存柊

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


