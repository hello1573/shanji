#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
山谷和山脊提取程序（简化版）
从 LAS 点云文件中提取山谷线和山脊线，映射回原始点云坐标
"""

import os
import sys
import yaml
import argparse
import copy
import shutil
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import laspy
from laspy.header import Version
import rasterio
from rasterio import features as rio_features
from rasterio.transform import Affine
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import morphology, measure
from skimage.graph import route_through_array
from shapely.geometry import LineString, box, Point
from shapely.ops import unary_union, linemerge, snap
import json
from pyproj import CRS
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image
from openness_ridge import extract_openness_ridges


DEFAULT_CONFIG_YAML = r"""
input_las: "D:/xishudimiandian/shanjixiantiqu/data"
output_dir: "outputs_flowtrace_v13"
ground_class: 2
ground_classes: [2, 16]

simple_openness:
  enabled: true
  classes: [2, 16]
  output_class: 3
  point_buffer_distance: 10.0
  point_mapping_mode: "nearest_line_cell"
  points_per_line_cell: 3
  radius_m: 120.0
  tpi_radius_m: 120.0
  sample_step_m: 6.0
  directions: 8
  smooth_sigma_cells: 0.9
  render_percentile_low: 2.0
  render_percentile_high: 98.0
  render_gamma: 1.0
  render_invert: false
  valley_openness_percentile: 35.0
  ridge_openness_percentile: 70.0
  valley_tpi_percentile: 35.0
  ridge_tpi_percentile: 65.0
  closing_disk: 1
  min_area_cells: 20
  line_min_cells: 8

dtm:
  resolution: 2.0
  max_fill_distance: 20.0
  smooth_sigma_cells: 1.2

ridge_dtm:
  smooth_sigma_cells: 0.9

hydrology:
  fill_sinks: false

extraction:
  method: "flow_trace_two_stage"
  min_accumulation_cells: 30

valley:
  primary:
    seed_percentile: 97.8
    continue_percentile: 85.0
    min_line_length: 90.0
    keep_top_n: 30
    max_dedup_seeds: 800
  supplement:
    enabled: true
    seed_percentile: 95.8
    continue_percentile: 78.0
    min_line_length: 70.0
    keep_top_n: 25
    max_dedup_seeds: 500

ridge:
  primary:
    seed_percentile: 98.5
    continue_percentile: 93.0
    min_line_length: 90.0
    keep_top_n: 0
    max_dedup_seeds: 150
    min_seed_distance: 35.0
  supplement:
    enabled: false
    seed_percentile: 96.0
    continue_percentile: 90.0
    min_line_length: 80.0
    keep_top_n: 8
    max_dedup_seeds: 120
    min_seed_distance: 35.0

trace:
  max_gap_cells: 5
  min_seed_distance: 35.0
  max_dedup_seeds: 1000
  allow_connect_to_existing: true
  visited_overlap_threshold: 0.55

supplement_filter:
  min_distance_from_existing: 30.0
  near_ratio_threshold: 0.75
  sample_points: 30
  spatial_grid_size: 180.0
  keep_per_grid: 3

ridge_supplement_filter:
  min_distance_from_existing: 10.0
  near_ratio_threshold: 0.94
  sample_points: 30
  spatial_grid_size: 120.0
  keep_per_grid: 6

local_supplement:
  enabled: true
  ridge_enabled: false
  grid_size: 220.0
  valley_local_seed_percentile: 90.0
  ridge_local_seed_percentile: 97.0
  max_seeds_per_grid: 1
  only_far_from_existing: true
  min_distance_from_existing: 24.0

edge_filter:
  enabled: true
  ridge_edge_buffer_m: 80.0
  valley_edge_buffer_m: 20.0
  max_near_edge_ratio: 0.10
  reject_if_endpoint_near_edge: true
  endpoint_edge_buffer_m: 70.0
  short_line_length_m: 120.0
  short_line_near_edge_ratio: 0.05

line_prune:
  enabled: true
  valley_min_distance: 18.0
  ridge_min_distance: 14.0
  near_ratio_threshold: 0.70

postprocess:
  merge_distance: 12.0
  simplify_tolerance: 0.8
  max_merge_angle_deg: 35.0
  max_merge_iterations: 3

postprocess_valley:
  merge_distance: 12.0
  simplify_tolerance: 0.8
  max_merge_angle_deg: 35.0
  max_merge_iterations: 3

postprocess_ridge:
  merge_distance: 20.0
  simplify_tolerance: 0.8
  max_merge_angle_deg: 50.0
  max_merge_iterations: 4

point_mapping:
  point_buffer_distance: 5.0
  use_importance_filter: true
  min_importance_level: "medium"

output:
  save_feature_points: true
  save_final_preview: true
  save_stage_previews: false
  save_debug_images: false
  save_seed_debug: false
  save_broad_debug: false
  save_ridge_debug: false
  save_openness_debug: false
  debug_outputs: false

broad_valley:
  enabled: true
  radius_m: 120.0
  valley_score_percentile: 80.0
  min_area_cells: 100
  min_line_length: 80.0
  max_slope_deg: 30.0
  keep_top_n: 30
  min_distance_from_existing: 25.0
  near_ratio_threshold: 0.75
  filter_existing_flow_lines: false
  merge_to_valley: true
  closing_disk: 4

broad_valley_prune:
  min_distance: 12.0
  near_ratio_threshold: 0.85

major_valley_filter:
  enabled: true
  min_line_length_m: 130.0
  min_mean_acc_percentile: 60.0
  min_valley_extreme_ratio: 0.40
  min_valley_relief_m: 1.0
  keep_top_n: 80
  use_for_ridge: true
  use_for_output: false

terrain_active:
  enabled: true
  relief_radius_m: [80.0, 160.0]
  min_relief_m: [1.0, 1.5]

ridge_openness_walk:
  smooth_iters: 2
  smooth_k: 1

  # multi-scale background scales as [broad_iters, broad_k] pairs.
  # bigger pairs capture broad dominant ridges; small pairs capture sharp crests.
  bg_scales: [[6, 2], [15, 3], [40, 4]]

  # adaptive thresholds (percentiles of each tile's positive strength)
  seed_pct: 90
  prom_continue_pct: 65
  min_mean_prom_pct: 80

  # absolute floors = the noise gate.
  prom_seed: 0.9
  prom_continue: 0.6
  min_mean_prom: 1.0

  # geometry / selection
  min_length_cells: 120
  prune_spur_cells: 8
  keep_top_n: 30

profile_ridge:
  sample_distances_m: [20.0, 40.0, 80.0, 160.0, 240.0]
  directions: 8
  shoulder_factor: 0.50

broad_crest_ridge:
  enabled: true
  radii_m: [80.0, 160.0, 240.0]
  profile_distances_m: [40.0, 80.0, 160.0, 240.0]
  directions: 8
  profile_weight: 0.35
  openness_weight: 0.25
  tpi_weight: 0.20
  relief_weight: 0.20
  score_percentile: 72.0
  min_tpi_percentile: 28.0
  min_local_relief_m: 0.8
  local_max_window_m: 140.0
  local_max_tolerance: 0.08
  min_distance_to_valley_m: 6.0
  min_edge_distance_m: 70.0
  min_distance_from_existing_ridge_m: 35.0
  closing_disk: 2
  min_area_cells: 60
  min_line_length: 120.0
  keep_top_n: 60

watershed_divide_ridge:
  enabled: true
  min_valley_line_length_m: 140.0
  major_valley_keep_top_n: 45
  include_broad_valley: false
  use_broad_valley_for_measure: true
  min_basin_area_cells: 500
  max_unassigned_fill_distance_m: 120.0
  min_distance_to_valley_m: 16.0
  min_hand_m: 1.8
  tpi_radius_m: 180.0
  min_tpi_percentile: 35.0
  min_edge_distance_m: 70.0
  boundary_dilation_m: 8.0
  closing_disk: 1
  min_area_cells: 60
  min_line_length: 130.0
  keep_top_n: 40
  save_debug: false

ridge_center_supplement:
  enabled: true
  local_max_window_m: 160.0
  local_max_tolerance_m: 5.0
  min_distance_to_valley_m: 18.0
  min_hand_m: 1.8
  tpi_radius_m: 180.0
  min_tpi_percentile: 38.0
  min_edge_distance_m: 70.0
  min_distance_from_existing_ridge_m: 45.0
  closing_disk: 1
  min_area_cells: 70
  min_line_length: 130.0
  keep_top_n: 20
  save_debug: false

ridge_final_filter:
  enabled: true
  min_line_length_m: 100.0
  min_mean_score: 0.38
  profile_half_width_m: 45.0
  min_profile_extreme_ratio: 0.22
  min_profile_relief_m: 0.5
  min_distance_to_valley_m: 8.0
  max_near_valley_ratio: 0.35
  min_edge_distance_m: 70.0

broad_crest_final_filter:
  min_line_length_m: 120.0
  min_mean_score: 0.36
  min_profile_extreme_ratio: 0.10
  min_profile_relief_m: 0.7
  max_near_valley_ratio: 0.35
  min_edge_distance_m: 70.0

connector_final_filter:
  enabled: true
  min_line_length_m: 20.0
  min_mean_score: 0.40
  min_profile_extreme_ratio: 0.0
  min_profile_relief_m: 0.0
  min_distance_to_valley_m: 6.0
  max_near_valley_ratio: 0.20
  min_edge_distance_m: 70.0

ridge_dense_prune:
  enabled: true
  min_distance_m: 20.0
  near_ratio_threshold: 0.60
  max_angle_diff_deg: 25.0

broad_ridge:
  enabled: false
  radius_m: 120.0
  ridge_score_percentile: 68.0
  min_area_cells: 70
  min_line_length: 80.0
  keep_top_n: 50
  closing_disk: 2
  use_curvature: true
  curvature_weight: 0.20
  tpi_weight: 1.45
  slope_weight: 0.05
  suppress_flow: false
  filter_existing_flow_lines: false
  min_distance_from_existing: 20.0
  near_ratio_threshold: 0.75

broad_ridge_prune:
  min_distance: 12.0
  near_ratio_threshold: 0.80

ridge_openness:
  enabled: false
  radii_m: [60.0, 120.0, 200.0]
  sample_step_m: 6.0
  directions: 8
  openness_weight: 0.45
  tpi_weight: 0.45
  curvature_weight: 0.10
  slope_weight: 0.0
  score_percentile: 68.0
  min_tpi_percentile: 40.0
  max_slope_deg: 60.0
  min_distance_to_valley: 10.0
  closing_disk: 2
  min_area_cells: 50
  min_line_length: 90.0
  keep_top_n: 30
  profile_half_width_m: 45.0
  ridge_extreme_ratio_min: 0.05
  ridge_relief_min: 0.05
  filter_existing_ridge: false
  min_distance_from_existing: 12.0
  near_ratio_threshold: 0.85
  suppress_flow: false
  suppress_flow_distance: 30.0
  suppress_flow_near_ratio: 0.40

ridge_divide_axis:
  enabled: false
  min_distance_to_valley: 10.0
  distance_percentile: 35.0
  tpi_radius_m: 180.0
  min_tpi_percentile: 28.0
  max_slope_deg: 65.0
  distance_weight: 0.15
  tpi_weight: 0.45
  curvature_weight: 0.05
  openness_weight: 0.15
  relief_weight: 0.25
  openness_radius_m: 180.0
  openness_sample_step_m: 8.0
  local_max_window_m: 60.0
  local_max_tolerance: 0.12
  closing_disk: 2
  min_area_cells: 30
  min_line_length: 70.0
  keep_top_n: 100
  profile_half_width_m: 50.0
  ridge_extreme_ratio_min: 0.02
  ridge_relief_min: 0.02
  filter_existing_ridge: false
  min_distance_from_existing: 18.0
  near_ratio_threshold: 0.85

ridge_divide_axis_prune:
  min_distance: 12.0
  near_ratio_threshold: 0.80

ridge_gap_connect:
  enabled: true
  min_gap_m: 12.0
  max_gap_m: 180.0
  max_angle_deg: 65.0
  min_mean_score: 0.42
  min_min_score: 0.18
  max_path_factor: 1.70
  max_near_valley_ratio: 0.25
  min_valley_distance_m: 6.0
  min_edge_distance_m: 70.0
  ridge_score_weight: 0.55
  profile_score_weight: 0.25
  tpi_weight: 0.15
  valley_distance_weight: 0.05
  search_margin_m: 60.0
  max_connections_per_iter: 80
  iterations: 1
  min_connector_length_m: 12.0
  max_connector_length_m: 180.0
  network_snap_tolerance_m: 8.0
  network_min_length_m: 120.0

valley_divide_ridge:
  enabled: false
  local_max_window_m: 90.0
  local_max_tolerance_m: 10.0
  min_distance_to_valley_m: 18.0
  min_edge_distance_m: 50.0
  tpi_radius_m: 180.0
  min_tpi_percentile: 32.0
  min_relief_to_valley_m: 1.5
  closing_disk: 2
  min_area_cells: 40
  min_line_length: 90.0
  keep_top_n: 120

ridge_shoulder:
  enabled: false
  radius_m: 18.0
  ridge_score_percentile: 79.0
  min_area_cells: 15
  min_line_length: 16.0
  keep_top_n: 90
  closing_disk: 1
  use_curvature: true
  curvature_weight: 1.35
  tpi_weight: 0.75
  slope_weight: 0.35
  filter_existing_flow_lines: false

ridge_shoulder_prune:
  min_distance: 6.0
  near_ratio_threshold: 0.94

line_importance:
  enabled: true
  profile_half_width_m: 45.0
  n_line_samples: 30
  n_profile_samples: 21
  valley_extreme_ratio_min: 0.60
  valley_relief_min: 2.0
  ridge_extreme_ratio_min: 0.25
  ridge_relief_min: 0.5
  high_threshold: 0.70
  medium_threshold: 0.32

dual_source:
  enabled: false
  candidate_source:
    enabled: true
    classes: [2, 1]
    exclude_classes: [4, 16]
    smooth_sigma_cells: 1.0
    prefer_stable_ground_in_mixed_cell: true
    stable_ground_class: 2
    candidate_class: 1
  pass1_source:
    enabled: true
    classes: [2, 16]
    exclude_classes: [1, 4]
    smooth_sigma_cells: 1.0

structure_fusion:
  enabled: true
  ridge_buffer_m: 15.0
  valley_buffer_m: 18.0
  conflict_distance_m: 10.0
  output_distance_maps: true
  output_debug_preview: false
  output_point_las: false
  point_include_classes: [2, 16]
  point_exclude_classes: [1, 4]

grid:
  use_unified_bounds: true
  bounds_source: "las_header"
"""


def deep_update(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_default_config() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_YAML)


class TerrainAnalyzer:
    """山谷和山脊提取分析类"""

    def __init__(self, config: dict, input_las: str, output_dir: str):
        """初始化单个点云任务的配置和输出目录。"""
        self.config = copy.deepcopy(config)

        self.input_las = str(input_las)
        self.output_dir = str(output_dir)

        self.config['input_las'] = self.input_las
        self.config['output_dir'] = self.output_dir
        ground_classes_cfg = self.config.get('ground_classes')
        if ground_classes_cfg is None:
            ground_classes_cfg = self.config.get('simple_openness', {}).get(
                'classes',
                [self.config.get('ground_class', 2)]
            )
        self.ground_classes = [int(v) for v in ground_classes_cfg]
        self.ground_class = self.ground_classes[0] if self.ground_classes else int(self.config.get('ground_class', 2))

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.las_all = None
        self.crs = None
        self._dtm_override = None
        self.source_class_usage = {}

        self.ground_points = self.read_ground_points()
        print(f"[✓] 读取地面点：{len(self.ground_points)} 个")

    def read_ground_points(self) -> np.ndarray:
        """
        读取 LAS 文件，筛选 classification=2 的地面点
        返回 Nx3 数组 [x, y, z]
        """
        las = laspy.read(self.input_las)
        self.las_all = las
        
        # 获取坐标系
        if hasattr(las, 'header') and hasattr(las.header, 'crs'):
            self.crs = las.header.crs
        
        # 筛选地面点
        cls = np.asarray(las.classification)
        mask = np.isin(cls, self.ground_classes)
        points = np.column_stack([
            las.x[mask],
            las.y[mask],
            las.z[mask]
        ])
        return points

    def build_dtm(self) -> Tuple[np.ndarray, Affine, Tuple[int, int], np.ndarray]:
        """
        从地面点生成 DTM，使用 median 而非 max
        返回 (dtm_array, transform, shape, raw_valid_mask)
        """
        if self._dtm_override is not None:
            dtm, transform, shape, raw_valid_mask = self._dtm_override
            return dtm.copy(), transform, shape, raw_valid_mask.copy()

        resolution = self.config['dtm']['resolution']
        
        # 计算栅格范围
        min_x = self.ground_points[:, 0].min()
        max_x = self.ground_points[:, 0].max()
        min_y = self.ground_points[:, 1].min()
        max_y = self.ground_points[:, 1].max()
        
        cols = int(np.ceil((max_x - min_x) / resolution)) + 1
        rows = int(np.ceil((max_y - min_y) / resolution)) + 1
        
        # 初始化 DTM
        dtm = np.full((rows, cols), np.nan, dtype=np.float32)
        raw_valid_mask = np.zeros((rows, cols), dtype=bool)
        
        # 点转栅格坐标
        col_idx = ((self.ground_points[:, 0] - min_x) / resolution).astype(int)
        row_idx = ((max_y - self.ground_points[:, 1]) / resolution).astype(int)
        z_vals = self.ground_points[:, 2]
        
        # 保留有效索引
        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        col_idx = col_idx[valid]
        row_idx = row_idx[valid]
        z_vals = z_vals[valid]
        
        flat_idx = row_idx.astype(np.int64) * cols + col_idx.astype(np.int64)
        order = np.argsort(flat_idx, kind='mergesort')
        flat_sorted = flat_idx[order]
        z_sorted = z_vals[order]

        unique_flat, start_idx, counts = np.unique(
            flat_sorted,
            return_index=True,
            return_counts=True
        )
        medians = np.empty(len(unique_flat), dtype=np.float32)
        for k, (start, count) in enumerate(zip(start_idx, counts)):
            medians[k] = np.median(z_sorted[start:start + count])

        rr = unique_flat // cols
        cc = unique_flat % cols
        dtm[rr, cc] = medians
        raw_valid_mask[rr, cc] = True
        
        # 创建地理仿射变换
        transform = Affine.translation(min_x, max_y) * Affine.scale(resolution, -resolution)
        
        print(f"[✓] 生成 DTM：{rows} x {cols}，分辨率 {resolution}m，有效格网 {np.sum(raw_valid_mask)}")
        return dtm, transform, (rows, cols), raw_valid_mask

    def _ensure_las_loaded(self):
        if self.las_all is None:
            self.las_all = laspy.read(self.input_las)
            if hasattr(self.las_all, 'header') and hasattr(self.las_all.header, 'crs'):
                self.crs = self.las_all.header.crs
        return self.las_all

    def make_output_las_header(self, las) -> laspy.LasHeader:
        header = las.header.copy()
        if header.version < Version(1, 1):
            header.version = Version(1, 1)
        return header

    def read_points_by_classes(self, classes, exclude_classes=None) -> np.ndarray:
        """Read LAS points by classification. Returns Nx4 [x, y, z, class]."""
        las = self._ensure_las_loaded()
        classes = set(int(v) for v in (classes or []))
        exclude_classes = set(int(v) for v in (exclude_classes or []))

        cls = np.asarray(las.classification)
        mask = np.isin(cls, list(classes)) if classes else np.ones(len(cls), dtype=bool)
        if exclude_classes:
            mask &= ~np.isin(cls, list(exclude_classes))

        if int(np.sum(mask)) == 0:
            print(f"[!] warning: no LAS points for classes={sorted(classes)} exclude={sorted(exclude_classes)}")
            return np.empty((0, 4), dtype=np.float64)

        return np.column_stack([
            np.asarray(las.x)[mask],
            np.asarray(las.y)[mask],
            np.asarray(las.z)[mask],
            cls[mask].astype(np.float64)
        ])

    def get_unified_grid_bounds(self, las=None, exclude_classes=None) -> dict:
        las = las or self._ensure_las_loaded()
        grid_cfg = self.config.get("grid", {})
        resolution = float(self.config.get("dtm", {}).get("resolution", 2.0))
        bounds_source = grid_cfg.get("bounds_source", "las_header")

        if bounds_source == "config_bounds" and grid_cfg.get("bounds"):
            b = grid_cfg["bounds"]
            min_x = float(b["min_x"])
            max_x = float(b["max_x"])
            min_y = float(b["min_y"])
            max_y = float(b["max_y"])
        elif bounds_source == "non_excluded_points":
            cls = np.asarray(las.classification)
            exclude_classes = set(int(v) for v in (exclude_classes or [4]))
            mask = ~np.isin(cls, list(exclude_classes))
            if int(np.sum(mask)) == 0:
                print("[!] warning: empty non-excluded bounds; falling back to LAS header")
                min_x, min_y, _ = las.header.mins
                max_x, max_y, _ = las.header.maxs
            else:
                xs = np.asarray(las.x)[mask]
                ys = np.asarray(las.y)[mask]
                min_x, max_x = float(xs.min()), float(xs.max())
                min_y, max_y = float(ys.min()), float(ys.max())
        else:
            min_x, min_y, _ = las.header.mins
            max_x, max_y, _ = las.header.maxs

        cols = int(np.ceil((max_x - min_x) / resolution)) + 1
        rows = int(np.ceil((max_y - min_y) / resolution)) + 1
        transform = Affine.translation(min_x, max_y) * Affine.scale(resolution, -resolution)
        return {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
            "rows": rows,
            "cols": cols,
            "shape": (rows, cols),
            "transform": transform,
            "resolution": resolution,
        }

    def build_dtm_from_points(self, points: np.ndarray, source_name: str,
                              unified_grid: dict, class_labels=None):
        """Build a median DTM on a shared grid, with optional class-aware selection."""
        rows = int(unified_grid["rows"])
        cols = int(unified_grid["cols"])
        min_x = float(unified_grid["min_x"])
        max_y = float(unified_grid["max_y"])
        resolution = float(unified_grid["resolution"])
        transform = unified_grid["transform"]

        dtm = np.full((rows, cols), np.nan, dtype=np.float32)
        raw_valid_mask = np.zeros((rows, cols), dtype=bool)
        usage = {
            "candidate_class1_total": 0,
            "candidate_class1_used_count": 0,
            "candidate_class1_grid_count": 0,
            "candidate_class2_grid_count": 0,
            "candidate_mixed_grid_count": 0,
        }
        class1_used_mask = np.zeros((rows, cols), dtype=np.uint8)

        if points is None or len(points) == 0:
            print(f"[!] warning: source {source_name} has no points; DTM will be empty")
            usage["candidate_class1_used_mask"] = class1_used_mask
            self.source_class_usage[source_name] = usage
            return dtm, transform, (rows, cols), raw_valid_mask

        points = np.asarray(points)
        col_idx = ((points[:, 0] - min_x) / resolution).astype(np.int64)
        row_idx = ((max_y - points[:, 1]) / resolution).astype(np.int64)
        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        if int(np.sum(valid)) == 0:
            print(f"[!] warning: source {source_name} has no points inside unified grid")
            usage["candidate_class1_used_mask"] = class1_used_mask
            self.source_class_usage[source_name] = usage
            return dtm, transform, (rows, cols), raw_valid_mask

        col_idx = col_idx[valid]
        row_idx = row_idx[valid]
        z_vals = points[valid, 2]
        if class_labels is not None:
            labels = np.asarray(class_labels)[valid].astype(np.int32)
        elif points.shape[1] >= 4:
            labels = points[valid, 3].astype(np.int32)
        else:
            labels = None

        flat_idx = row_idx * cols + col_idx
        source_cfg = self.config.get("dual_source", {}).get(source_name, {})
        prefer_mixed = bool(source_cfg.get("prefer_stable_ground_in_mixed_cell", False))
        stable_class = int(source_cfg.get("stable_ground_class", 2))
        candidate_class = int(source_cfg.get("candidate_class", 1))

        if labels is not None and prefer_mixed:
            usage["candidate_class1_total"] = int(np.sum(labels == candidate_class))
            class1_flats = np.unique(flat_idx[labels == candidate_class])
            class2_flats = np.unique(flat_idx[labels == stable_class])
            class1_set = set(class1_flats.tolist())
            class2_set = set(class2_flats.tolist())
            class1_only_set = class1_set - class2_set
            usage["candidate_class1_grid_count"] = int(len(class1_set))
            usage["candidate_class2_grid_count"] = int(len(class2_set))
            usage["candidate_mixed_grid_count"] = int(len(class1_set & class2_set))

            class1_only_mask = np.isin(flat_idx, list(class1_only_set)) if class1_only_set else np.zeros_like(flat_idx, dtype=bool)
            stable_mask = labels == stable_class
            candidate_used_mask = (labels == candidate_class) & class1_only_mask
            usage["candidate_class1_used_count"] = int(np.sum(candidate_used_mask))
            selected = stable_mask | candidate_used_mask
            class1_used_flat = np.unique(flat_idx[candidate_used_mask])
            if len(class1_used_flat):
                class1_used_mask[class1_used_flat // cols, class1_used_flat % cols] = 1
            flat_idx = flat_idx[selected]
            z_vals = z_vals[selected]

        if len(flat_idx) == 0:
            usage["candidate_class1_used_mask"] = class1_used_mask
            self.source_class_usage[source_name] = usage
            return dtm, transform, (rows, cols), raw_valid_mask

        order = np.argsort(flat_idx, kind='mergesort')
        flat_sorted = flat_idx[order]
        z_sorted = z_vals[order]
        unique_flat, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
        medians = np.empty(len(unique_flat), dtype=np.float32)
        for k, (start, count) in enumerate(zip(start_idx, counts)):
            medians[k] = np.median(z_sorted[start:start + count])

        rr = unique_flat // cols
        cc = unique_flat % cols
        dtm[rr, cc] = medians
        raw_valid_mask[rr, cc] = True
        usage["candidate_class1_used_mask"] = class1_used_mask
        self.source_class_usage[source_name] = usage
        print(f"[{source_name}] DTM grid={rows}x{cols}, valid_cells={int(np.sum(raw_valid_mask))}")
        return dtm, transform, (rows, cols), raw_valid_mask

    def fill_nodata_limited(self, dtm: np.ndarray, raw_valid_mask: np.ndarray,
                            max_fill_distance: float, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        有限距离填补 NaN 值
        返回 (filled_dtm, support_mask)
        support_mask=1 表示有可靠点云支撑
        """
        filled = dtm.copy()
        valid_mask = raw_valid_mask.astype(bool)

        if np.sum(valid_mask) == 0:
            print("[!] 警告：没有有效地面点，无法填补 NaN")
            return filled, valid_mask

        # 基于有效像元构建 KDTree
        valid_rc = np.column_stack(np.where(valid_mask))
        valid_xy = np.column_stack([valid_rc[:, 1] * resolution, valid_rc[:, 0] * resolution])
        valid_z = dtm[valid_rc[:, 0], valid_rc[:, 1]]

        tree = cKDTree(valid_xy)

        # 待填补像元
        fill_rc = np.column_stack(np.where(~valid_mask & np.isnan(dtm)))
        if fill_rc.size == 0:
            return filled, valid_mask

        fill_xy = np.column_stack([fill_rc[:, 1] * resolution, fill_rc[:, 0] * resolution])

        k = min(8, len(valid_xy))
        if k == 0:
            return filled, valid_mask

        dists, idxs = tree.query(fill_xy, k=k, distance_upper_bound=max_fill_distance)

        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # 计算 IDW
        p = 2.0
        eps = 1e-6
        invalid = idxs >= len(valid_z)
        safe_idxs = np.where(invalid, 0, idxs)
        neighbor_z = valid_z[safe_idxs]
        weights = 1.0 / (dists ** p + eps)
        weights[~np.isfinite(dists)] = 0.0
        weights[invalid] = 0.0

        weight_sum = np.sum(weights, axis=1)
        ok = weight_sum > 0
        filled_vals = np.zeros(len(fill_rc), dtype=np.float32)
        filled_vals[ok] = (np.sum(neighbor_z * weights, axis=1)[ok] / weight_sum[ok]).astype(np.float32)

        fill_mask = np.zeros_like(valid_mask, dtype=bool)
        fill_mask[fill_rc[:, 0], fill_rc[:, 1]] = ok
        filled[fill_rc[ok, 0], fill_rc[ok, 1]] = filled_vals[ok]

        # 创建 support_mask：原始有效区 + 有限距离填补区
        support_mask = valid_mask | fill_mask

        print(f"[✓] IDW 填补：支撑区域 {np.sum(support_mask)} 像元")
        return filled, support_mask

    def fill_sinks(self, dtm: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
        """
        填洼处理，用于水文分析前的预处理
        只在 support_mask=True 的区域内处理
        """
        # 当前关闭填洼，保持原始 DTM 结构
        return dtm.copy()

    def nan_safe_gaussian_smooth(self, dtm: np.ndarray, support_mask: np.ndarray, sigma: float) -> np.ndarray:
        """
        NaN 安全的高斯平滑，仅在 support_mask 内进行
        """
        valid = (~np.isnan(dtm)) & (support_mask > 0)

        values = np.where(valid, dtm, 0.0).astype(float)
        weights = valid.astype(float)

        smooth_values = gaussian_filter(values, sigma=sigma)
        smooth_weights = gaussian_filter(weights, sigma=sigma)

        out = np.full_like(dtm, np.nan, dtype=np.float32)
        ok = smooth_weights > 1e-6
        out[ok] = smooth_values[ok] / smooth_weights[ok]
        out[support_mask == 0] = np.nan
        return out

    def compute_flow_direction(self, dtm: np.ndarray, support_mask: np.ndarray, 
                              resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        D8 流向计算
        返回 (flow_to_r, flow_to_c) - 指向下游像元的行列坐标
        无下坡邻居时，值为 -1
        """
        rows, cols = dtm.shape
        flow_to_r = np.full((rows, cols), -1, dtype=np.int32)
        flow_to_c = np.full((rows, cols), -1, dtype=np.int32)
        
        drows = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
        dcols = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
        distances = np.array([
            resolution,
            resolution * np.sqrt(2),
            resolution,
            resolution * np.sqrt(2),
            resolution,
            resolution * np.sqrt(2),
            resolution,
            resolution * np.sqrt(2)
        ], dtype=np.float32)

        if rows > 2 and cols > 2:
            center_slice = (slice(1, rows - 1), slice(1, cols - 1))
            center_valid = (
                (support_mask[center_slice] > 0)
                & np.isfinite(dtm[center_slice])
            )
            best_slope = np.zeros((rows - 2, cols - 2), dtype=np.float32)
            r_grid, c_grid = np.indices((rows - 2, cols - 2), dtype=np.int32)
            r_grid += 1
            c_grid += 1

            flow_to_r_view = flow_to_r[center_slice]
            flow_to_c_view = flow_to_c[center_slice]

            for dr, dc, dist in zip(drows, dcols, distances):
                neighbor_slice = (
                    slice(1 + int(dr), rows - 1 + int(dr)),
                    slice(1 + int(dc), cols - 1 + int(dc))
                )
                neighbor_valid = (
                    (support_mask[neighbor_slice] > 0)
                    & np.isfinite(dtm[neighbor_slice])
                )
                slope = (dtm[center_slice] - dtm[neighbor_slice]) / dist
                update = center_valid & neighbor_valid & (slope > best_slope)
                best_slope[update] = slope[update]
                flow_to_r_view[update] = (r_grid + dr)[update]
                flow_to_c_view[update] = (c_grid + dc)[update]
        
        print(f"[✓] D8 流向计算完成")
        return flow_to_r, flow_to_c

    def compute_flow_accumulation(self, dtm: np.ndarray, flow_to_r: np.ndarray, 
                                  flow_to_c: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
        """
        汇流累积计算 - 使用拓扑排序
        返回累积数组
        """
        rows, cols = dtm.shape
        valid = (support_mask > 0) & np.isfinite(dtm)
        accumulation = valid.astype(np.float32)
        
        # 计算每个像元的入流数（有多少个上游像元指向它）
        inflow_count = np.zeros((rows, cols), dtype=np.int32)
        edge_mask = valid & (flow_to_r >= 0) & (flow_to_c >= 0)
        if np.any(edge_mask):
            np.add.at(
                inflow_count,
                (flow_to_r[edge_mask], flow_to_c[edge_mask]),
                1
            )
        
        # BFS 拓扑排序：从入流数为 0 的像元开始处理
        from collections import deque
        queue = deque(zip(*np.where(valid & (inflow_count == 0))))
        
        # 处理队列
        while queue:
            i, j = queue.popleft()
            
            # 如果有下游像元，将累积量加到下游
            if flow_to_r[i, j] >= 0:
                ni, nj = flow_to_r[i, j], flow_to_c[i, j]
                accumulation[ni, nj] += accumulation[i, j]
                inflow_count[ni, nj] -= 1
                
                # 如果下游像元的入流数变为 0，加入队列
                if inflow_count[ni, nj] == 0:
                    queue.append((ni, nj))
        
        # support_mask=False 或 NaN 的地方累积量为 0
        accumulation[~valid] = 0
        
        print(f"[✓] 汇流累积计算完成（拓扑排序）")
        return accumulation

    def extract_valley_lines(self, accumulation: np.ndarray,
                             support_mask: np.ndarray) -> np.ndarray:
        """
        提取山谷线（汇流累积高的区域）
        只在 support_mask=True 的区域内计算阈值
        返回二值化的山谷掩膜
        """
        # 只在支持区域内计算阈值
        valid_acc = accumulation[support_mask > 0]
        valid_acc = valid_acc[valid_acc > 0]
        
        if len(valid_acc) == 0:
            print("[!] 警告：山谷提取失败，无有效汇流值")
            return np.zeros_like(accumulation, dtype=np.uint8)
        
        percentile = self.config.get('extraction', {}).get('accumulation_percentile')
        if percentile is None:
            valley_cfg = self.config.get('valley', {})
            primary_cfg = valley_cfg.get('primary', valley_cfg)
            percentile = primary_cfg.get('continue_percentile', 90.0)
        min_cells = self.config.get('extraction', {}).get('min_accumulation_cells', 10)
        threshold = max(np.percentile(valid_acc, percentile), min_cells)
        
        # 创建山谷掩膜
        valley_mask = (accumulation >= threshold) & (support_mask > 0)

        # 轻微闭运算，连接断裂区域
        valley_mask = morphology.binary_closing(valley_mask, morphology.disk(1))
        valley_mask = valley_mask & (support_mask > 0)

        # 最小化处理（去小碎片）
        valley_mask = morphology.remove_small_objects(valley_mask, min_size=min_cells)
        
        print(f"[✓] 提取山谷线掩膜（阈值：{threshold:.1f}，像元：{np.sum(valley_mask)}）")
        return valley_mask.astype(np.uint8)

    def extract_ridge_lines(self, dtm: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
        """
        提取山脊线（反地形法）
        返回二值化的山脊掩膜
        """
        # 反地形处理
        max_z = np.nanmax(dtm[support_mask > 0])
        inv_dtm = max_z - dtm
        
        # 对反地形进行填洼
        filled_inv = self.fill_sinks(inv_dtm, support_mask)
        
        # 计算反地形的流向和汇流
        flow_to_r_inv, flow_to_c_inv = self.compute_flow_direction(filled_inv, support_mask, 
                                                                     self.config['dtm']['resolution'])
        accumulation_inv = self.compute_flow_accumulation(filled_inv, flow_to_r_inv, flow_to_c_inv, support_mask)
        
        # 提取高汇流区域（只在支持区域内计算阈值）
        valid_acc = accumulation_inv[support_mask > 0]
        valid_acc = valid_acc[valid_acc > 0]
        
        if len(valid_acc) == 0:
            print("[!] 警告：山脊提取失败，无有效汇流值")
            return np.zeros_like(dtm, dtype=np.uint8)
        
        percentile = self.config.get('extraction', {}).get('accumulation_percentile')
        if percentile is None:
            ridge_cfg = self.config.get('ridge', {})
            primary_cfg = ridge_cfg.get('primary', ridge_cfg)
            percentile = primary_cfg.get('continue_percentile', 90.0)
        min_cells = self.config.get('extraction', {}).get('min_accumulation_cells', 10)
        threshold = max(np.percentile(valid_acc, percentile), min_cells)
        
        ridge_mask = (accumulation_inv >= threshold) & (support_mask > 0)

        # 轻微闭运算，连接断裂区域
        ridge_mask = morphology.binary_closing(ridge_mask, morphology.disk(1))
        ridge_mask = ridge_mask & (support_mask > 0)

        # 最小化处理
        ridge_mask = morphology.remove_small_objects(ridge_mask, min_size=min_cells)
        
        print(f"[✓] 提取山脊线掩膜（阈值：{threshold:.1f}，像元：{np.sum(ridge_mask)}）")
        return ridge_mask.astype(np.uint8)

    def build_accumulation_mask(self, accumulation: np.ndarray, support_mask: np.ndarray,
                                percentile: float, min_cells: int) -> Tuple[np.ndarray, float]:
        """
        用于调试输出的掩膜构建
        """
        valid_acc = accumulation[(support_mask > 0) & (accumulation > 0)]
        if len(valid_acc) == 0:
            return np.zeros_like(accumulation, dtype=np.uint8), np.nan

        threshold = np.percentile(valid_acc, percentile)
        mask = (accumulation >= threshold) & (support_mask > 0)
        mask = morphology.binary_closing(mask, morphology.disk(1))
        mask = mask & (support_mask > 0)
        mask = morphology.remove_small_objects(mask, min_size=min_cells)
        return mask.astype(np.uint8), float(threshold)

    def build_upstream_index(
        self,
        flow_to_r: np.ndarray,
        flow_to_c: np.ndarray,
        support_mask: np.ndarray
    ) -> Dict[int, List[Tuple[int, int]]]:
        rows, cols = support_mask.shape
        edge_mask = (support_mask > 0) & (flow_to_r >= 0) & (flow_to_c >= 0)
        if not np.any(edge_mask):
            return {}

        src_r, src_c = np.where(edge_mask)
        dst_r = flow_to_r[edge_mask]
        dst_c = flow_to_c[edge_mask]

        in_bounds = (
            (dst_r >= 0)
            & (dst_c >= 0)
            & (dst_r < rows)
            & (dst_c < cols)
        )
        dst_valid = np.zeros_like(in_bounds, dtype=bool)
        if np.any(in_bounds):
            dst_valid[in_bounds] = support_mask[dst_r[in_bounds], dst_c[in_bounds]] > 0
        if not np.any(dst_valid):
            return {}

        src_r = src_r[dst_valid]
        src_c = src_c[dst_valid]
        dst_r = dst_r[dst_valid]
        dst_c = dst_c[dst_valid]

        dst_flat = dst_r.astype(np.int64) * cols + dst_c.astype(np.int64)
        order = np.argsort(dst_flat, kind='mergesort')
        dst_flat = dst_flat[order]
        src_r = src_r[order]
        src_c = src_c[order]

        unique_flat, start_idx, counts = np.unique(
            dst_flat,
            return_index=True,
            return_counts=True
        )

        upstream = {}
        for flat, start, count in zip(unique_flat, start_idx, counts):
            upstream[int(flat)] = [
                (int(r), int(c))
                for r, c in zip(src_r[start:start + count], src_c[start:start + count])
            ]

        return upstream

    def trace_main_flow_lines(self, accumulation: np.ndarray, flow_to_r: np.ndarray, flow_to_c: np.ndarray,
                              support_mask: np.ndarray, transform: Affine,
                              seed_percentile: float, continue_percentile: float,
                              min_length: float, keep_top_n: int,
                              existing_visited: Optional[np.ndarray] = None,
                              feature_type: str = "unknown",
                              external_seeds: Optional[List[Tuple[int, int]]] = None,
                              max_dedup_seeds: int = 0,
                              min_seed_distance_override: Optional[float] = None,
                              upstream: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> List[LineString]:
        """
        基于双阈值的主线追踪
        """
        valid_acc = accumulation[(support_mask > 0) & (accumulation > 0)]
        if len(valid_acc) == 0:
            return []

        continue_threshold = np.percentile(valid_acc, continue_percentile)

        trace_cfg = self.config.get('trace', {})
        max_gap_cells = int(trace_cfg.get('max_gap_cells', 0))
        if min_seed_distance_override is not None:
            min_seed_distance = float(min_seed_distance_override)
        else:
            min_seed_distance = float(trace_cfg.get('min_seed_distance', 0.0))
        allow_connect_to_existing = bool(trace_cfg.get('allow_connect_to_existing', True))
        visited_overlap_threshold = float(trace_cfg.get('visited_overlap_threshold', 0.7))
        is_supplement = existing_visited is not None

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        min_seed_distance_cells = min_seed_distance / resolution if min_seed_distance > 0 else 0.0
        min_seed_distance_sq = min_seed_distance_cells * min_seed_distance_cells

        rows, cols = accumulation.shape

        if upstream is None:
            upstream = self.build_upstream_index(flow_to_r, flow_to_c, support_mask)

        if external_seeds is not None:
            selected_seeds = list(external_seeds)
            seed_total = len(selected_seeds)
            seed_threshold = 0.0
        else:
            seed_threshold = np.percentile(valid_acc, seed_percentile)
            seeds = np.column_stack(np.where((support_mask > 0) & (accumulation >= seed_threshold)))
            if seeds.size == 0:
                return []

            seed_total = len(seeds)
            seed_values = accumulation[seeds[:, 0], seeds[:, 1]]
            order = np.argsort(-seed_values)

            dedup_limit = max_dedup_seeds if max_dedup_seeds > 0 else len(order)
            selected_seeds = []
            for idx in order:
                if len(selected_seeds) >= dedup_limit:
                    break
                r, c = seeds[idx]
                if min_seed_distance_cells > 0 and selected_seeds:
                    too_close = False
                    for sr, sc in selected_seeds:
                        if (r - sr) ** 2 + (c - sc) ** 2 < min_seed_distance_sq:
                            too_close = True
                            break
                    if too_close:
                        continue
                selected_seeds.append((r, c))

        seeds_dedup = len(selected_seeds)
        visited = np.zeros_like(support_mask, dtype=bool)
        max_steps = rows * cols
        lines_with_score = []
        candidate_count = 0
        length_count = 0

        for r, c in selected_seeds:
            local_seen = {(r, c)}

            # 上游主支追踪
            upstream_path = []
            gap_count = 0
            gap_start = None
            steps = 0
            ur, uc = r, c

            while True:
                key = ur * cols + uc
                candidates = upstream.get(key, [])
                if not candidates:
                    break

                best = None
                best_acc = -1.0
                for cr, cc in candidates:
                    if support_mask[cr, cc] == 0:
                        continue
                    if (cr, cc) in local_seen:
                        continue
                    block_visited = existing_visited[cr, cc] if is_supplement else visited[cr, cc]
                    if not allow_connect_to_existing and block_visited:
                        continue
                    acc_val = accumulation[cr, cc]
                    if acc_val >= continue_threshold and acc_val > best_acc:
                        best = (cr, cc)
                        best_acc = acc_val

                if best is None:
                    if max_gap_cells <= 0:
                        break
                    for cr, cc in candidates:
                        if support_mask[cr, cc] == 0:
                            continue
                        if (cr, cc) in local_seen:
                            continue
                        block_visited = existing_visited[cr, cc] if is_supplement else visited[cr, cc]
                        if not allow_connect_to_existing and block_visited:
                            continue
                        acc_val = accumulation[cr, cc]
                        if acc_val > best_acc:
                            best = (cr, cc)
                            best_acc = acc_val

                if best is None:
                    break

                acc_val = accumulation[best]
                if acc_val < continue_threshold:
                    gap_count += 1
                    if gap_start is None:
                        gap_start = len(upstream_path)
                else:
                    gap_count = 0
                    gap_start = None

                upstream_path.append(best)
                local_seen.add(best)

                connect_visited = existing_visited[best] if is_supplement else visited[best]
                if connect_visited and allow_connect_to_existing:
                    break

                steps += 1
                if gap_count > max_gap_cells:
                    if gap_start is not None:
                        upstream_path = upstream_path[:gap_start]
                    break
                if steps >= max_steps:
                    break

                ur, uc = best

            if gap_count > 0 and gap_start is not None:
                upstream_path = upstream_path[:gap_start]

            # 下游追踪
            downstream_path = []
            gap_count = 0
            gap_start = None
            steps = 0
            dr, dc = r, c

            while True:
                nr = flow_to_r[dr, dc]
                nc = flow_to_c[dr, dc]
                if nr < 0 or nc < 0:
                    break
                if support_mask[nr, nc] == 0:
                    break
                if accumulation[nr, nc] <= 0:
                    break
                block_visited = existing_visited[nr, nc] if is_supplement else visited[nr, nc]
                if not allow_connect_to_existing and block_visited:
                    break

                acc_val = accumulation[nr, nc]
                if acc_val < continue_threshold:
                    gap_count += 1
                    if gap_start is None:
                        gap_start = len(downstream_path)
                else:
                    gap_count = 0
                    gap_start = None

                downstream_path.append((nr, nc))
                local_seen.add((nr, nc))

                connect_visited = existing_visited[(nr, nc)] if is_supplement else visited[nr, nc]
                if connect_visited and allow_connect_to_existing:
                    break

                steps += 1
                if gap_count > max_gap_cells:
                    if gap_start is not None:
                        downstream_path = downstream_path[:gap_start]
                    break
                if steps >= max_steps:
                    break

                dr, dc = nr, nc

            if gap_count > 0 and gap_start is not None:
                downstream_path = downstream_path[:gap_start]

            path = list(reversed(upstream_path)) + [(r, c)] + downstream_path
            if len(path) < 2:
                continue

            candidate_count += 1

            visited_count = 0
            for pr, pc in path:
                if visited[pr, pc]:
                    visited_count += 1
            visited_ratio = visited_count / len(path)
            if visited_ratio > visited_overlap_threshold:
                continue

            coords = []
            max_acc = 0.0
            for pr, pc in path:
                x = transform.c + (pc + 0.5) * transform.a
                y = transform.f + (pr + 0.5) * transform.e
                coords.append([x, y])
                if accumulation[pr, pc] > max_acc:
                    max_acc = accumulation[pr, pc]

            line = LineString(coords)
            if line.length < min_length:
                continue

            length_count += 1
            lines_with_score.append((line, max_acc, line.length))
            for pr, pc in path:
                visited[pr, pc] = True

        spatial_cfg = self.config.get('spatial_balance', {})
        spatial_enabled = bool(spatial_cfg.get('enabled', False))
        balanced_count = len(lines_with_score)

        if spatial_enabled:
            grid_size = float(spatial_cfg.get('grid_size', 0.0))
            keep_per_grid = int(spatial_cfg.get('keep_per_grid', 1))
            lines_with_score = self.spatial_balance_select(
                lines_with_score,
                grid_size=grid_size,
                keep_per_grid=keep_per_grid,
                keep_top_n=keep_top_n
            )
            balanced_count = len(lines_with_score)
        else:
            lines_with_score.sort(key=lambda item: (item[1], item[2]), reverse=True)
            if keep_top_n is not None and keep_top_n > 0:
                lines_with_score = lines_with_score[:keep_top_n]

        final_count = len(lines_with_score)

        print(
            f"[trace-{feature_type}] seed_threshold={seed_threshold:.2f}, continue_threshold={continue_threshold:.2f}, "
            f"seeds_total={seed_total}, seeds_dedup={seeds_dedup}, "
            f"candidates={candidate_count}, length_filtered={length_count}, "
            f"spatial_balance={balanced_count}, keep_top_n={final_count}"
        )

        return [item[0] for item in lines_with_score]

    def spatial_balance_select(self, lines_with_score: List[Tuple[LineString, float, float]],
                               grid_size: float, keep_per_grid: int, keep_top_n: int) -> List[Tuple[LineString, float, float]]:
        """
        空间均衡选择
        """
        if not lines_with_score:
            return []

        if grid_size <= 0:
            lines_with_score.sort(key=lambda item: (item[1], item[2]), reverse=True)
            if keep_top_n is not None and keep_top_n > 0:
                return lines_with_score[:keep_top_n]
            return lines_with_score

        buckets = {}
        for line, max_acc, length in lines_with_score:
            centroid = line.centroid
            gx = int(centroid.x // grid_size)
            gy = int(centroid.y // grid_size)
            buckets.setdefault((gx, gy), []).append((line, max_acc, length))

        selected = []
        for key, items in buckets.items():
            items.sort(key=lambda item: (item[1], item[2]), reverse=True)
            selected.extend(items[:max(1, keep_per_grid)])

        selected.sort(key=lambda item: (item[1], item[2]), reverse=True)
        if keep_top_n is not None and keep_top_n > 0:
            selected = selected[:keep_top_n]
        return selected

    def filter_supplement_lines(self, supplement_lines: List[LineString],
                                existing_lines: List[LineString],
                                min_distance_from_existing: float,
                                near_ratio_threshold: float,
                                spatial_grid_size: float,
                                keep_per_grid: int) -> List[LineString]:
        if not supplement_lines:
            return []

        if not existing_lines:
            filtered = list(supplement_lines)
        else:
            existing_union = unary_union(existing_lines)
            filtered = []
            for line in supplement_lines:
                length = line.length
                if length == 0:
                    if line.distance(existing_union) >= min_distance_from_existing:
                        filtered.append(line)
                    continue

                step = max(1.0, length / 20.0)
                n_samples = max(2, int(length / step) + 1)
                distances = []
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    distances.append(pt.distance(existing_union))

                near_count = sum(1 for d in distances if d < min_distance_from_existing)
                ratio = near_count / len(distances)

                if ratio <= near_ratio_threshold:
                    filtered.append(line)

        if not filtered:
            return []

        if spatial_grid_size <= 0:
            return filtered

        buckets = {}
        for line in filtered:
            centroid = line.centroid
            gx = int(centroid.x // spatial_grid_size)
            gy = int(centroid.y // spatial_grid_size)
            buckets.setdefault((gx, gy), []).append(line)

        result = []
        for items in buckets.values():
            result.extend(items[:max(1, keep_per_grid)])

        return result

    def generate_local_supplement_seeds(self, accumulation: np.ndarray,
                                        support_mask: np.ndarray,
                                        existing_lines: List[LineString],
                                        transform: Affine,
                                        local_cfg: dict,
                                        feature_type: str) -> List[Tuple[int, int]]:
        grid_size = float(local_cfg.get('grid_size', 220.0))
        max_seeds_per = int(local_cfg.get('max_seeds_per_grid', 3))
        only_far = bool(local_cfg.get('only_far_from_existing', True))
        min_dist = float(local_cfg.get('min_distance_from_existing', 50.0))

        if feature_type == 'valley':
            local_p = float(local_cfg.get('valley_local_seed_percentile', 90.0))
        else:
            local_p = float(local_cfg.get('ridge_local_seed_percentile', 92.0))

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = accumulation.shape

        existing_union = unary_union(existing_lines) if (only_far and existing_lines) else None

        col_coords = np.arange(cols) * resolution + transform.c
        row_coords = transform.f + np.arange(rows) * transform.e
        col_bins = (col_coords - col_coords.min()) // grid_size
        row_bins = (row_coords - row_coords.min()) // grid_size

        all_seeds = []
        for rb in range(int(row_bins.max()) + 1):
            for cb in range(int(col_bins.max()) + 1):
                r_mask = row_bins == rb
                c_mask = col_bins == cb
                r_indices = np.where(r_mask)[0]
                c_indices = np.where(c_mask)[0]

                if len(r_indices) == 0 or len(c_indices) == 0:
                    continue

                sub_acc = accumulation[np.ix_(r_indices, c_indices)]
                sub_sup = support_mask[np.ix_(r_indices, c_indices)]

                valid = (sub_sup > 0) & (sub_acc > 0)
                if np.sum(valid) == 0:
                    continue

                valid_vals = sub_acc[valid]
                if len(valid_vals) < 5:
                    continue

                local_threshold = np.percentile(valid_vals, local_p)
                candidate_mask = (sub_acc >= local_threshold) & valid

                cand_rc = np.column_stack(np.where(candidate_mask))
                if len(cand_rc) == 0:
                    continue

                cand_acc = sub_acc[candidate_mask]
                order = np.argsort(-cand_acc)

                count = 0
                for idx in order:
                    if count >= max_seeds_per:
                        break
                    lr, lc = cand_rc[idx]
                    gr = r_indices[lr]
                    gc = c_indices[lc]

                    if only_far and existing_union is not None:
                        x = transform.c + (gc + 0.5) * transform.a
                        y = transform.f + (gr + 0.5) * transform.e
                        pt = Point(x, y)
                        if pt.distance(existing_union) < min_dist:
                            continue

                    all_seeds.append((int(gr), int(gc)))
                    count += 1

        print(f"[local-supplement-{feature_type}] generated {len(all_seeds)} seeds")
        return all_seeds

    def prune_dense_lines(self, lines: List[LineString], min_distance: float,
                          near_ratio_threshold: float) -> List[LineString]:
        if not lines or len(lines) <= 1:
            return lines

        scored = []
        for line in lines:
            scored.append((line, line.length))
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = []
        selected_union = None

        for line, length in scored:
            if selected_union is None:
                selected.append(line)
                selected_union = line
                continue

            n_samples = 30
            distances = []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                distances.append(pt.distance(selected_union))

            near_count = sum(1 for d in distances if d < min_distance)
            ratio = near_count / len(distances)

            if ratio <= near_ratio_threshold:
                selected.append(line)
                selected_union = unary_union(selected)

        return selected

    def remove_flow_lines_near_broad_valley(self, flow_lines: List[LineString],
                                            broad_lines: List[LineString],
                                            min_distance: float = 30.0,
                                            near_ratio_threshold: float = 0.6) -> List[LineString]:
        if not broad_lines or not flow_lines:
            return flow_lines

        broad_union = unary_union(broad_lines)
        kept = []
        for line in flow_lines:
            n_samples = 30
            near_count = 0
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                if pt.distance(broad_union) < min_distance:
                    near_count += 1
            ratio = near_count / n_samples
            if ratio <= near_ratio_threshold:
                kept.append(line)
        return kept

    def remove_lines_near_lines(self, candidate_lines: List[LineString],
                                reference_lines: List[LineString],
                                min_distance: float = 14.0,
                                near_ratio_threshold: float = 0.75) -> List[LineString]:
        if not candidate_lines or not reference_lines:
            return candidate_lines
        ref_union = unary_union(reference_lines)
        kept = []
        for line in candidate_lines:
            n_samples = 30
            near_count = 0
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                if pt.distance(ref_union) < min_distance:
                    near_count += 1
            ratio = near_count / n_samples
            if ratio <= near_ratio_threshold:
                kept.append(line)
        return kept

    def filter_lines_by_importance(self, lines: List[LineString],
                                   importance: Optional[List[dict]],
                                   min_level: str = 'medium') -> List[LineString]:
        if not importance:
            return lines
        rank = {'low': 0, 'medium': 1, 'high': 2}
        min_rank = rank.get(min_level, 1)
        kept = []
        for line, imp in zip(lines, importance):
            level = imp.get('importance_level', 'low')
            if rank.get(level, 0) >= min_rank:
                kept.append(line)
        return kept

    def extract_two_stage_lines(self, accumulation: np.ndarray,
                                flow_to_r: np.ndarray, flow_to_c: np.ndarray,
                                support_mask: np.ndarray, transform: Affine,
                                primary_cfg: dict, supplement_cfg: dict,
                                feature_type: str) -> List[LineString]:
        upstream = self.build_upstream_index(flow_to_r, flow_to_c, support_mask)
        primary_lines = self.trace_main_flow_lines(
            accumulation, flow_to_r, flow_to_c, support_mask, transform,
            seed_percentile=primary_cfg['seed_percentile'],
            continue_percentile=primary_cfg['continue_percentile'],
            min_length=primary_cfg['min_line_length'],
            keep_top_n=primary_cfg['keep_top_n'],
            feature_type=f"{feature_type}-primary",
            max_dedup_seeds=primary_cfg.get('max_dedup_seeds', 0),
            min_seed_distance_override=primary_cfg.get('min_seed_distance', None),
            upstream=upstream
        )
        print(f"[primary-{feature_type}] lines={len(primary_lines)}")

        all_existing = list(primary_lines)

        if supplement_cfg.get('enabled', False):
            supplement_candidates = self.trace_main_flow_lines(
                accumulation, flow_to_r, flow_to_c, support_mask, transform,
                seed_percentile=supplement_cfg['seed_percentile'],
                continue_percentile=supplement_cfg['continue_percentile'],
                min_length=supplement_cfg['min_line_length'],
                keep_top_n=supplement_cfg['keep_top_n'],
                feature_type=f"{feature_type}-supplement",
                max_dedup_seeds=supplement_cfg.get('max_dedup_seeds', 0),
                min_seed_distance_override=supplement_cfg.get('min_seed_distance', None),
                upstream=upstream
            )
            print(f"[supplement-{feature_type}] candidates={len(supplement_candidates)}")

            if feature_type == "ridge":
                filter_cfg = self.config.get('ridge_supplement_filter', self.config.get('supplement_filter', {}))
            else:
                filter_cfg = self.config.get('supplement_filter', {})
            min_dist = filter_cfg.get('min_distance_from_existing', 30.0)
            near_threshold = filter_cfg.get('near_ratio_threshold', 0.75)
            grid_size = filter_cfg.get('spatial_grid_size', 180.0)
            keep_per = filter_cfg.get('keep_per_grid', 3)

            after_near = self.filter_supplement_lines(
                supplement_candidates, primary_lines, min_dist, near_threshold, 0, 0
            )
            print(f"[supplement-{feature_type}] after_near_ratio_filter={len(after_near)}")

            after_spatial = self.filter_supplement_lines(
                after_near, [], 0, 0, grid_size, keep_per
            )
            print(f"[supplement-{feature_type}] after_spatial_filter={len(after_spatial)}")

            all_existing = primary_lines + after_spatial

        local_cfg = self.config.get('local_supplement', {})
        local_enabled = local_cfg.get('enabled', False)
        if feature_type == 'ridge':
            local_enabled = local_enabled and local_cfg.get('ridge_enabled', True)
        if local_enabled:
            local_seeds = self.generate_local_supplement_seeds(
                accumulation, support_mask, all_existing, transform,
                local_cfg, feature_type
            )

            if local_seeds:
                local_lines = self.trace_main_flow_lines(
                    accumulation, flow_to_r, flow_to_c, support_mask, transform,
                    seed_percentile=50.0,
                    continue_percentile=supplement_cfg.get('continue_percentile', 72.0),
                    min_length=supplement_cfg.get('min_line_length', 35.0),
                    keep_top_n=100,
                    feature_type=f"{feature_type}-local",
                    external_seeds=local_seeds,
                    max_dedup_seeds=0,
                    upstream=upstream
                )
                print(f"[local-{feature_type}] traced {len(local_lines)} lines")

                if local_lines:
                    if feature_type == "ridge":
                        local_filter_cfg = self.config.get('ridge_supplement_filter', self.config.get('supplement_filter', {}))
                    else:
                        local_filter_cfg = self.config.get('supplement_filter', {})
                    local_min_dist = local_filter_cfg.get('min_distance_from_existing', 30.0)
                    local_near = local_filter_cfg.get('near_ratio_threshold', 0.75)
                    local_grid = local_filter_cfg.get('spatial_grid_size', 180.0)
                    local_keep = local_filter_cfg.get('keep_per_grid', 3)

                    local_filtered = self.filter_supplement_lines(
                        local_lines, all_existing, local_min_dist, local_near, 0, 0
                    )
                    local_filtered = self.filter_supplement_lines(
                        local_filtered, [], 0, 0, local_grid, local_keep
                    )
                    print(f"[local-{feature_type}] after_filter={len(local_filtered)}")
                    all_existing = all_existing + local_filtered

        final_lines = all_existing
        print(f"[final-{feature_type}] lines={len(final_lines)}")
        return final_lines

    def save_debug_trace_seeds(self, accumulation: np.ndarray, accumulation_inv: np.ndarray,
                               support_mask: np.ndarray, dtm: np.ndarray,
                               transform: Affine):
        valley_cfg = self.config.get('valley', {})
        ridge_cfg = self.config.get('ridge', {})

        valid_acc = accumulation[(support_mask > 0) & (accumulation > 0)]
        valid_acc_inv = accumulation_inv[(support_mask > 0) & (accumulation_inv > 0)]

        if len(valid_acc) == 0 or len(valid_acc_inv) == 0:
            return

        vp = valley_cfg.get('primary', valley_cfg)
        vs = valley_cfg.get('supplement', valley_cfg)
        rp = ridge_cfg.get('primary', ridge_cfg)
        rs = ridge_cfg.get('supplement', ridge_cfg)

        vp_seed = np.percentile(valid_acc, vp.get('seed_percentile', 97.0))
        vs_seed = np.percentile(valid_acc, vs.get('seed_percentile', 94.5))
        rp_seed = np.percentile(valid_acc_inv, rp.get('seed_percentile', 98.0))
        rs_seed = np.percentile(valid_acc_inv, rs.get('seed_percentile', 95.5))

        vp_mask = (support_mask > 0) & (accumulation >= vp_seed)
        vs_mask = (support_mask > 0) & (accumulation >= vs_seed) & ~vp_mask
        rp_mask = (support_mask > 0) & (accumulation_inv >= rp_seed)
        rs_mask = (support_mask > 0) & (accumulation_inv >= rs_seed) & ~rp_mask

        rows, cols = dtm.shape
        fig, ax = plt.subplots(figsize=(14, 12), dpi=100)

        dtm_vis = dtm.copy()
        dtm_vis[support_mask == 0] = np.nan
        with np.errstate(invalid='ignore'):
            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.hillshade(dtm_vis, vert_exag=0.1)
        shaded[support_mask == 0] = 1.0
        ax.imshow(shaded, cmap='gray', vmin=0, vmax=1)

        vp_coords = np.column_stack(np.where(vp_mask))
        vs_coords = np.column_stack(np.where(vs_mask))
        rp_coords = np.column_stack(np.where(rp_mask))
        rs_coords = np.column_stack(np.where(rs_mask))

        if len(vp_coords) > 0:
            ax.scatter(vp_coords[:, 1], vp_coords[:, 0], c='darkblue', s=2, label=f'Valley primary seeds ({len(vp_coords)})')
        if len(vs_coords) > 0:
            ax.scatter(vs_coords[:, 1], vs_coords[:, 0], c='cornflowerblue', s=2, label=f'Valley supplement seeds ({len(vs_coords)})')
        if len(rp_coords) > 0:
            ax.scatter(rp_coords[:, 1], rp_coords[:, 0], c='darkred', s=2, label=f'Ridge primary seeds ({len(rp_coords)})')
        if len(rs_coords) > 0:
            ax.scatter(rs_coords[:, 1], rs_coords[:, 0], c='orangered', s=2, label=f'Ridge supplement seeds ({len(rs_coords)})')

        ax.set_title('Trace Seeds (primary vs supplement)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(False)

        output_path = os.path.join(self.output_dir, 'debug_trace_seeds.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[✓] 保存种子调试图：{output_path}")

    def compute_tpi_score(self, dtm: np.ndarray, support_mask: np.ndarray,
                          resolution: float, radius_m: float) -> np.ndarray:
        radius_cells = max(1, int(round(radius_m / resolution)))
        sigma = radius_cells / 2.0
        dtm_filled = np.where(np.isnan(dtm), 0.0, dtm)
        valid = (~np.isnan(dtm)) & (support_mask > 0)
        weights = valid.astype(float)

        smooth_dtm = gaussian_filter(dtm_filled, sigma=sigma)
        smooth_weights = gaussian_filter(weights, sigma=sigma)

        local_mean = np.where(smooth_weights > 1e-6, smooth_dtm / smooth_weights, np.nan)
        tpi = dtm - local_mean
        broad_valley_score = -tpi
        broad_valley_score[~valid] = np.nan
        return broad_valley_score

    def compute_slope(self, dtm: np.ndarray, resolution: float) -> np.ndarray:
        dx = np.gradient(dtm, resolution, axis=1)
        dy = np.gradient(dtm, resolution, axis=0)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        return slope_deg

    def compute_tpi_raw(self, dtm: np.ndarray, support_mask: np.ndarray,
                        resolution: float, radius_m: float) -> np.ndarray:
        radius_cells = max(1, int(round(radius_m / resolution)))
        sigma = radius_cells / 2.0
        dtm_filled = np.where(np.isnan(dtm), 0.0, dtm)
        valid = (~np.isnan(dtm)) & (support_mask > 0)
        weights = valid.astype(float)
        smooth_dtm = gaussian_filter(dtm_filled, sigma=sigma)
        smooth_weights = gaussian_filter(weights, sigma=sigma)
        local_mean = np.where(smooth_weights > 1e-6, smooth_dtm / smooth_weights, np.nan)
        tpi = dtm - local_mean
        tpi[~valid] = np.nan
        return tpi

    def compute_curvature_score(self, dtm: np.ndarray, support_mask: np.ndarray,
                                resolution: float) -> np.ndarray:
        valid = (~np.isnan(dtm)) & (support_mask > 0)
        dtm_filled = np.where(np.isnan(dtm), 0.0, dtm)
        sigma = 1.5
        smooth_dtm = gaussian_filter(dtm_filled, sigma=sigma)
        dxx = np.gradient(np.gradient(smooth_dtm, resolution, axis=1), resolution, axis=1)
        dyy = np.gradient(np.gradient(smooth_dtm, resolution, axis=0), resolution, axis=0)
        laplacian = dxx + dyy
        curvature_score = -laplacian
        curvature_score[~valid] = np.nan
        return curvature_score

    def robust_normalize(self, arr: np.ndarray, valid_mask: np.ndarray,
                         p_low: float = 10, p_high: float = 90) -> np.ndarray:
        vals = arr[valid_mask & np.isfinite(arr)]
        out = np.zeros_like(arr, dtype=np.float32)
        if len(vals) == 0:
            out[~valid_mask] = np.nan
            return out
        lo = np.percentile(vals, p_low)
        hi = np.percentile(vals, p_high)
        spread = hi - lo
        if spread < 1e-6:
            out[~valid_mask] = np.nan
            return out
        out = (arr - lo) / spread
        out = np.clip(out, 0, 1).astype(np.float32)
        out[~valid_mask] = np.nan
        return out

    def sample_offset(self, arr: np.ndarray, dr: int, dc: int,
                      fill=np.nan) -> np.ndarray:
        out = np.full(arr.shape, fill, dtype=np.float32)
        rows, cols = arr.shape

        if dr >= 0:
            src_r0, src_r1 = dr, rows
            dst_r0, dst_r1 = 0, rows - dr
        else:
            src_r0, src_r1 = 0, rows + dr
            dst_r0, dst_r1 = -dr, rows

        if dc >= 0:
            src_c0, src_c1 = dc, cols
            dst_c0, dst_c1 = 0, cols - dc
        else:
            src_c0, src_c1 = 0, cols + dc
            dst_c0, dst_c1 = -dc, cols

        if src_r1 > src_r0 and src_c1 > src_c0:
            out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]

        return out

    def compute_positive_openness(self, dtm: np.ndarray, support_mask: np.ndarray,
                                  resolution: float, radius_m: float,
                                  sample_step_m: float = 6.0,
                                  directions: int = 8) -> np.ndarray:
        valid = (support_mask > 0) & np.isfinite(dtm)
        rows, cols = dtm.shape

        radius_cells = max(1, int(round(radius_m / resolution)))
        step_cells = max(1, int(round(sample_step_m / resolution)))

        if directions == 8:
            dirs = [
                (-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1)
            ]
        else:
            dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        all_dir_openness = []

        for dr, dc in dirs:
            max_angle = np.full((rows, cols), -np.pi / 2, dtype=np.float32)

            for k in range(step_cells, radius_cells + 1, step_cells):
                nbr_z = self.sample_offset(dtm, dr * k, dc * k, fill=np.nan)
                nbr_valid = self.sample_offset(
                    valid.astype(np.float32), dr * k, dc * k, fill=0.0
                ) > 0.5

                dist_m = np.sqrt(
                    (dr * k * resolution) ** 2 + (dc * k * resolution) ** 2
                )
                angle = np.arctan2(nbr_z - dtm, dist_m)

                ok = valid & nbr_valid & np.isfinite(angle)
                max_angle[ok] = np.maximum(max_angle[ok], angle[ok])

            openness = 90.0 - np.degrees(max_angle)
            openness[~valid] = np.nan
            all_dir_openness.append(openness)

        positive_openness = np.nanmean(
            np.stack(all_dir_openness, axis=0), axis=0
        ).astype(np.float32)
        positive_openness[~valid] = np.nan
        return positive_openness

    def extract_broad_valley_lines(self, dtm: np.ndarray, support_mask: np.ndarray,
                                   transform: Affine,
                                   existing_valley_lines: List[LineString]) -> List[LineString]:
        bv_cfg = self.config.get('broad_valley', {})
        radius_m = float(bv_cfg.get('radius_m', 60.0))
        score_percentile = float(bv_cfg.get('valley_score_percentile', 88.0))
        min_area_cells = int(bv_cfg.get('min_area_cells', 80))
        min_line_length = float(bv_cfg.get('min_line_length', 120.0))
        max_slope_deg = float(bv_cfg.get('max_slope_deg', 35.0))
        min_dist_existing = float(bv_cfg.get('min_distance_from_existing', 40.0))
        near_ratio_thresh = float(bv_cfg.get('near_ratio_threshold', 0.8))
        keep_top_n = int(bv_cfg.get('keep_top_n', 12))

        resolution = abs(transform.a) if transform.a != 0 else 1.0

        score = self.compute_tpi_score(dtm, support_mask, resolution, radius_m)
        slope = self.compute_slope(dtm, resolution)

        valid_score = score[(support_mask > 0) & np.isfinite(score)]
        if len(valid_score) == 0:
            print("[!] broad_valley: 无有效 TPI 分数")
            return []

        threshold = np.percentile(valid_score, score_percentile)
        bv_mask = (score >= threshold) & (support_mask > 0) & np.isfinite(score) & (slope <= max_slope_deg)

        closing_disk = int(bv_cfg.get('closing_disk', 4))
        bv_mask = morphology.binary_closing(bv_mask, morphology.disk(closing_disk))
        bv_mask = bv_mask & (support_mask > 0)
        bv_mask = morphology.remove_small_objects(bv_mask, min_size=min_area_cells)

        bv_mask_uint8 = bv_mask.astype(np.uint8)
        skeleton = morphology.skeletonize(bv_mask_uint8 > 0)

        lines = self.skeleton_to_lines(bv_mask_uint8, transform, 'broad_valley', min_length=min_line_length)

        filter_existing = bool(bv_cfg.get('filter_existing_flow_lines', True))
        if filter_existing and existing_valley_lines:
            existing_union = unary_union(existing_valley_lines)
            filtered = []
            for line in lines:
                n_samples = 30
                near_count = 0
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    if pt.distance(existing_union) < min_dist_existing:
                        near_count += 1
                ratio = near_count / n_samples
                if ratio <= near_ratio_thresh:
                    filtered.append(line)
            lines = filtered

        if lines:
            scored_lines = []
            for line in lines:
                length = line.length
                n_samples = 30
                score_vals = []
                slope_vals = []
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    col = int((pt.x - transform.c) / transform.a)
                    row = int((pt.y - transform.f) / transform.e)
                    rows, cols = score.shape
                    if 0 <= row < rows and 0 <= col < cols:
                        s = score[row, col]
                        sl = slope[row, col]
                        if np.isfinite(s):
                            score_vals.append(s)
                        if np.isfinite(sl):
                            slope_vals.append(sl)
                mean_score = np.mean(score_vals) if score_vals else 0.0
                mean_slope = np.mean(slope_vals) if slope_vals else 1.0
                trunk_score = length * mean_score / (mean_slope + 1.0)
                scored_lines.append((line, trunk_score))

            scored_lines.sort(key=lambda x: x[1], reverse=True)
            lines = [item[0] for item in scored_lines[:keep_top_n]]

        print(f"[broad_valley] threshold={threshold:.2f}, mask_cells={np.sum(bv_mask)}, "
              f"after_score_filter={len(lines)} (keep_top_n={keep_top_n})")

        if self.config.get('output', {}).get('save_broad_debug', False):
            self._save_broad_valley_debug(dtm, support_mask, score, bv_mask, skeleton, transform)

        return lines

    def _save_broad_valley_debug(self, dtm: np.ndarray, support_mask: np.ndarray,
                                 score: np.ndarray, bv_mask: np.ndarray,
                                 skeleton: np.ndarray, transform: Affine):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        score_vis = score.copy()
        score_vis[~np.isfinite(score_vis)] = np.nan
        p_low = np.nanpercentile(score_vis, 5)
        p_high = np.nanpercentile(score_vis, 99)
        im = ax.imshow(np.clip(score_vis, p_low, p_high), cmap='YlOrRd')
        ax.set_title('Broad Valley Score (-TPI)')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_valley_score.png'), dpi=100, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(bv_mask.astype(np.uint8), cmap='Greens')
        ax.set_title('Broad Valley Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_valley_mask.png'), dpi=100, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(skeleton.astype(np.uint8), cmap='Purples')
        ax.set_title('Broad Valley Skeleton')
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_valley_skeleton.png'), dpi=100, bbox_inches='tight')
        plt.close()

        print(f"[✓] 保存 broad_valley 调试图（3 张）")

    def extract_broad_ridge_lines(self, dtm: np.ndarray, support_mask: np.ndarray,
                                  transform: Affine,
                                  existing_ridge_lines: List[LineString]) -> List[LineString]:
        br_cfg = self.config.get('broad_ridge', {})
        radius_m = float(br_cfg.get('radius_m', 60.0))
        score_percentile = float(br_cfg.get('ridge_score_percentile', 86.0))
        min_area_cells = int(br_cfg.get('min_area_cells', 40))
        min_line_length = float(br_cfg.get('min_line_length', 40.0))
        keep_top_n = int(br_cfg.get('keep_top_n', 60))
        closing_disk = int(br_cfg.get('closing_disk', 2))
        use_curvature = bool(br_cfg.get('use_curvature', True))
        curvature_w = float(br_cfg.get('curvature_weight', 0.7))
        tpi_w = float(br_cfg.get('tpi_weight', 1.0))
        slope_w = float(br_cfg.get('slope_weight', 0.3))

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, radius_m)
        slope = self.compute_slope(dtm, resolution)

        if use_curvature:
            curvature = self.compute_curvature_score(dtm, support_mask, resolution)

            tpi_valid = tpi[valid & np.isfinite(tpi)]
            curv_valid = curvature[valid & np.isfinite(curvature)]
            slope_valid = slope[valid & np.isfinite(slope)]

            if len(tpi_valid) == 0 or len(curv_valid) == 0:
                print("[!] broad_ridge: 无有效分数")
                return []

            def robust_normalize(arr, valid_mask):
                vals = arr[valid_mask & np.isfinite(arr)]
                if len(vals) == 0:
                    return np.zeros_like(arr)
                p10 = np.percentile(vals, 10)
                p90 = np.percentile(vals, 90)
                spread = p90 - p10
                if spread < 1e-6:
                    return np.zeros_like(arr)
                normed = (arr - p10) / spread
                return np.clip(normed, 0, 1)

            norm_tpi = robust_normalize(tpi, valid)
            norm_curv = robust_normalize(curvature, valid)
            norm_slope = robust_normalize(slope, valid)

            ridge_score = tpi_w * norm_tpi + curvature_w * norm_curv + slope_w * norm_slope
        else:
            ridge_score = tpi

        ridge_score[~valid] = np.nan
        valid_score = ridge_score[np.isfinite(ridge_score) & valid]
        if len(valid_score) == 0:
            print("[!] broad_ridge: 无有效 ridge_score")
            return []

        threshold = np.percentile(valid_score, score_percentile)
        br_mask = (ridge_score >= threshold) & valid & np.isfinite(ridge_score)

        br_mask = morphology.binary_closing(br_mask, morphology.disk(closing_disk))
        br_mask = br_mask & (support_mask > 0)
        br_mask = morphology.remove_small_objects(br_mask, min_size=min_area_cells)

        br_mask_uint8 = br_mask.astype(np.uint8)
        skeleton = morphology.skeletonize(br_mask_uint8 > 0)

        lines = self.skeleton_to_lines(br_mask_uint8, transform, 'broad_ridge', min_length=min_line_length)

        filter_existing = bool(br_cfg.get('filter_existing_flow_lines', True))
        if filter_existing and existing_ridge_lines:
            min_dist_existing = float(br_cfg.get('min_distance_from_existing', 18.0))
            near_ratio_thresh = float(br_cfg.get('near_ratio_threshold', 0.65))
            existing_union = unary_union(existing_ridge_lines)
            filtered = []
            for line in lines:
                n_samples = 30
                near_count = 0
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    if pt.distance(existing_union) < min_dist_existing:
                        near_count += 1
                near_ratio = near_count / n_samples
                if near_ratio <= near_ratio_thresh:
                    filtered.append(line)
            lines = filtered

        if lines:
            scored_lines = []
            rows_d, cols_d = ridge_score.shape
            for line in lines:
                length = line.length
                n_samples = 30
                score_vals = []
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    col = int((pt.x - transform.c) / transform.a)
                    row = int((pt.y - transform.f) / transform.e)
                    if 0 <= row < rows_d and 0 <= col < cols_d:
                        s = ridge_score[row, col]
                        if np.isfinite(s):
                            score_vals.append(s)
                mean_score = np.mean(score_vals) if score_vals else 0.0
                trunk_score = length * mean_score
                scored_lines.append((line, trunk_score))

            scored_lines.sort(key=lambda x: x[1], reverse=True)
            lines = [item[0] for item in scored_lines[:keep_top_n]]

        print(f"[broad_ridge] threshold={threshold:.2f}, mask_cells={np.sum(br_mask)}, "
              f"after_score_filter={len(lines)} (keep_top_n={keep_top_n})")

        if self.config.get('output', {}).get('save_broad_debug', False):
            self._save_broad_ridge_debug(dtm, support_mask, ridge_score, br_mask, skeleton, transform)

        return lines

    def _save_broad_ridge_debug(self, dtm: np.ndarray, support_mask: np.ndarray,
                                score: np.ndarray, br_mask: np.ndarray,
                                skeleton: np.ndarray, transform: Affine):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        score_vis = score.copy()
        score_vis[~np.isfinite(score_vis)] = np.nan
        p_low = np.nanpercentile(score_vis, 5)
        p_high = np.nanpercentile(score_vis, 99)
        im = ax.imshow(np.clip(score_vis, p_low, p_high), cmap='YlOrRd')
        ax.set_title('Broad Ridge Score (TPI)')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_ridge_score.png'), dpi=100, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(br_mask.astype(np.uint8), cmap='Oranges')
        ax.set_title('Broad Ridge Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_ridge_mask.png'), dpi=100, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(skeleton.astype(np.uint8), cmap='Reds')
        ax.set_title('Broad Ridge Skeleton')
        plt.savefig(os.path.join(self.output_dir, 'debug_broad_ridge_skeleton.png'), dpi=100, bbox_inches='tight')
        plt.close()

        print(f"[✓] 保存 broad_ridge 调试图（3 张）")

    def extract_ridge_shoulder_lines(self, dtm: np.ndarray, support_mask: np.ndarray,
                                     transform: Affine,
                                     existing_ridge_lines: List[LineString]) -> List[LineString]:
        cfg = self.config.get('ridge_shoulder', {})
        if not cfg.get('enabled', False):
            return []

        radius_m = float(cfg.get('radius_m', 18.0))
        score_percentile = float(cfg.get('ridge_score_percentile', 77.0))
        min_area_cells = int(cfg.get('min_area_cells', 15))
        min_line_length = float(cfg.get('min_line_length', 16.0))
        keep_top_n = int(cfg.get('keep_top_n', 130))
        closing_disk = int(cfg.get('closing_disk', 1))
        use_curvature = bool(cfg.get('use_curvature', True))
        curvature_w = float(cfg.get('curvature_weight', 1.35))
        tpi_w = float(cfg.get('tpi_weight', 0.75))
        slope_w = float(cfg.get('slope_weight', 0.35))

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, radius_m)
        slope = self.compute_slope(dtm, resolution)

        if use_curvature:
            curvature = self.compute_curvature_score(dtm, support_mask, resolution)

            def robust_normalize(arr, valid_mask):
                vals = arr[valid_mask & np.isfinite(arr)]
                if len(vals) == 0:
                    return np.zeros_like(arr)
                p10 = np.percentile(vals, 10)
                p90 = np.percentile(vals, 90)
                spread = p90 - p10
                if spread < 1e-6:
                    return np.zeros_like(arr)
                return np.clip((arr - p10) / spread, 0, 1)

            norm_tpi = robust_normalize(tpi, valid)
            norm_curv = robust_normalize(curvature, valid)
            norm_slope = robust_normalize(slope, valid)
            ridge_score = tpi_w * norm_tpi + curvature_w * norm_curv + slope_w * norm_slope
        else:
            ridge_score = tpi

        ridge_score[~valid] = np.nan
        valid_score = ridge_score[np.isfinite(ridge_score) & valid]
        if len(valid_score) == 0:
            print("[!] ridge_shoulder: 无有效 ridge_score")
            return []

        threshold = np.percentile(valid_score, score_percentile)
        mask = (ridge_score >= threshold) & valid & np.isfinite(ridge_score)

        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask = mask & (support_mask > 0)
        mask = morphology.remove_small_objects(mask, min_size=min_area_cells)

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            'ridge_shoulder',
            min_length=min_line_length
        )

        scored_lines = []
        rows_d, cols_d = ridge_score.shape
        for line in lines:
            if line.is_empty:
                continue
            length = line.length
            score_vals = []
            n_samples = 30
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                col = int((pt.x - transform.c) / transform.a)
                row = int((pt.y - transform.f) / transform.e)
                if 0 <= row < rows_d and 0 <= col < cols_d:
                    s = ridge_score[row, col]
                    if np.isfinite(s):
                        score_vals.append(s)
            mean_score = np.mean(score_vals) if score_vals else 0.0
            scored_lines.append((line, length * mean_score))

        scored_lines.sort(key=lambda x: x[1], reverse=True)
        result = [item[0] for item in scored_lines[:keep_top_n]]

        print(f"[ridge_shoulder] threshold={threshold:.2f}, mask_cells={np.sum(mask)}, lines={len(result)}")

        if self.config.get('output', {}).get('save_broad_debug', False):
            self._save_ridge_shoulder_debug(dtm, support_mask, ridge_score, mask, transform)

        return result

    def _save_ridge_shoulder_debug(self, dtm: np.ndarray, support_mask: np.ndarray,
                                   score: np.ndarray, mask: np.ndarray, transform: Affine):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        score_vis = score.copy()
        score_vis[~np.isfinite(score_vis)] = np.nan
        p_low = np.nanpercentile(score_vis, 5)
        p_high = np.nanpercentile(score_vis, 99)
        im = ax.imshow(np.clip(score_vis, p_low, p_high), cmap='YlOrRd')
        ax.set_title('Ridge Shoulder Score')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_ridge_shoulder_score.png'), dpi=100, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(mask.astype(np.uint8), cmap='Oranges')
        ax.set_title('Ridge Shoulder Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_ridge_shoulder_mask.png'), dpi=100, bbox_inches='tight')
        plt.close()

        print(f"[✓] 保存 ridge_shoulder 调试图（2 张）")

    def extract_openness_ridge_lines(self, dtm: np.ndarray, support_mask: np.ndarray,
                                     transform: Affine,
                                     valley_lines: List[LineString],
                                     existing_ridge_lines: List[LineString]) -> List[LineString]:
        cfg = self.config.get('ridge_openness', {})
        if not cfg.get('enabled', False):
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        radii_m = cfg.get('radii_m', [40.0, 80.0, 140.0])
        sample_step_m = float(cfg.get('sample_step_m', 6.0))
        directions = int(cfg.get('directions', 8))

        openness_maps = []
        for radius_m in radii_m:
            op = self.compute_positive_openness(
                dtm, support_mask, resolution,
                radius_m=float(radius_m),
                sample_step_m=sample_step_m,
                directions=directions
            )
            openness_maps.append(self.robust_normalize(op, valid))

        openness_score = np.nanmean(np.stack(openness_maps, axis=0), axis=0)

        tpi_radius = max(float(r) for r in radii_m)
        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, tpi_radius)
        curvature = self.compute_curvature_score(dtm, support_mask, resolution)
        slope = self.compute_slope(dtm, resolution)

        norm_tpi = self.robust_normalize(tpi, valid)
        norm_curv = self.robust_normalize(curvature, valid)

        openness_w = float(cfg.get('openness_weight', 0.55))
        tpi_w = float(cfg.get('tpi_weight', 0.35))
        curv_w = float(cfg.get('curvature_weight', 0.10))

        ridge_score = openness_w * openness_score + tpi_w * norm_tpi + curv_w * norm_curv
        ridge_score[~valid] = np.nan

        valid_score = ridge_score[valid & np.isfinite(ridge_score)]
        if len(valid_score) == 0:
            print("[!] ridge_openness: 无有效 score")
            return []

        score_threshold = np.percentile(valid_score, float(cfg.get('score_percentile', 78.0)))

        valid_tpi = tpi[valid & np.isfinite(tpi)]
        if len(valid_tpi) > 0:
            tpi_threshold = np.percentile(valid_tpi, float(cfg.get('min_tpi_percentile', 55.0)))
        else:
            tpi_threshold = 0.0

        max_slope_deg = float(cfg.get('max_slope_deg', 55.0))
        mask_before_valley_filter = (
            valid
            & np.isfinite(ridge_score)
            & (ridge_score >= score_threshold)
            & np.isfinite(tpi)
            & (tpi >= tpi_threshold)
            & np.isfinite(slope)
            & (slope <= max_slope_deg)
        )

        mask = mask_before_valley_filter.copy()
        valley_dist_arr = None

        min_distance_to_valley = float(cfg.get('min_distance_to_valley', 18.0))
        if valley_lines and min_distance_to_valley > 0:
            valley_dist_arr = self.compute_line_distance_grid(
                valley_lines, dtm.shape, transform, support_mask, resolution
            )
            mask = mask & (valley_dist_arr >= min_distance_to_valley)

        mask_after_valley_filter = mask.copy()

        closing_disk = int(cfg.get('closing_disk', 2))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask = mask & (support_mask > 0)
        mask = morphology.remove_small_objects(mask, min_size=int(cfg.get('min_area_cells', 80)))
        final_mask = mask.copy()

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            'ridge_openness',
            min_length=float(cfg.get('min_line_length', 100.0))
        )

        if not lines:
            print("[ridge_openness] lines=0")
            return []

        scored = []
        rows_d, cols_d = ridge_score.shape

        profile_hw = float(cfg.get('profile_half_width_m', 35.0))
        er_min = float(cfg.get('ridge_extreme_ratio_min', 0.45))
        relief_min = float(cfg.get('ridge_relief_min', 1.2))

        for line in lines:
            if line.is_empty:
                continue

            er, relief, vc = self.evaluate_line_extremeness(
                line, dtm, transform, resolution,
                mode='ridge',
                profile_half_width_m=profile_hw,
                n_line_samples=30,
                n_profile_samples=21
            )

            if er < er_min or relief < relief_min:
                continue

            score_vals = []
            n_samples = 30
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows_d and 0 <= c < cols_d:
                    s = ridge_score[r, c]
                    if np.isfinite(s):
                        score_vals.append(s)

            mean_score = np.mean(score_vals) if score_vals else 0.0
            final_score = line.length * mean_score * (0.5 + er) * (1.0 + min(relief, 10.0) / 10.0)
            scored.append((line, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_top_n = int(cfg.get('keep_top_n', 120))
        result = [x[0] for x in scored[:keep_top_n]]

        if cfg.get('filter_existing_ridge', True) and existing_ridge_lines and result:
            result = self.filter_supplement_lines(
                result,
                existing_ridge_lines,
                float(cfg.get('min_distance_from_existing', 12.0)),
                float(cfg.get('near_ratio_threshold', 0.85)),
                0,
                0
            )

        if self.config.get('output', {}).get('save_openness_debug', False):
            self._save_ridge_openness_full_debug(
                dtm, support_mask,
                openness_score, norm_tpi, norm_curv,
                ridge_score,
                mask_before_valley_filter,
                valley_dist_arr,
                mask_after_valley_filter,
                final_mask,
                transform
            )

        print(
            f"[ridge_openness] score_th={score_threshold:.3f}, tpi_th={tpi_threshold:.3f}, "
            f"mask_cells={int(np.sum(mask))}, lines={len(result)}"
        )

        return result

    def _save_ridge_openness_full_debug(
        self, dtm, support_mask,
        openness_score, norm_tpi, norm_curv,
        ridge_score,
        mask_before_valley_filter,
        valley_dist_arr,
        mask_after_valley_filter,
        final_mask,
        transform
    ):
        def _save_img(arr, title, fname, cmap='YlOrRd'):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            vis = arr.copy() if hasattr(arr, 'copy') else arr
            if vis.dtype == bool or vis.dtype == np.bool_:
                ax.imshow(vis.astype(np.uint8), cmap='Oranges')
            else:
                vis = vis.astype(float)
                vis[~np.isfinite(vis)] = np.nan
                finite_vals = vis[np.isfinite(vis)]
                if len(finite_vals) > 0:
                    p_low = np.nanpercentile(finite_vals, 5)
                    p_high = np.nanpercentile(finite_vals, 99)
                else:
                    p_low, p_high = 0, 1
                im = ax.imshow(np.clip(vis, p_low, p_high), cmap=cmap)
                plt.colorbar(im, ax=ax)
            ax.set_title(title)
            plt.savefig(os.path.join(self.output_dir, fname), dpi=100, bbox_inches='tight')
            plt.close()

        _save_img(openness_score, 'Openness Score (normalized avg)', 'debug_ro_openness_score.png')
        _save_img(norm_tpi, 'Norm TPI', 'debug_ro_norm_tpi.png', cmap='RdBu_r')
        _save_img(norm_curv, 'Norm Curvature', 'debug_ro_norm_curvature.png', cmap='RdBu_r')
        _save_img(ridge_score, 'Ridge Score (final)', 'debug_ro_ridge_score.png')
        _save_img(mask_before_valley_filter, 'Mask before valley filter', 'debug_ro_mask_before_valley.png')
        if valley_dist_arr is not None:
            _save_img(valley_dist_arr, 'Valley Distance (m)', 'debug_ro_valley_distance.png', cmap='Blues')
        _save_img(mask_after_valley_filter, 'Mask after valley filter', 'debug_ro_mask_after_valley.png')
        _save_img(final_mask, 'Final mask (after morph)', 'debug_ro_final_mask.png')
        print("[debug] saved ridge_openness full debug (8 images)")

    def _save_divide_axis_debug(self, divide_score, center_band, mask, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir

        def _save_img(arr, title, fname, cmap='YlOrRd'):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            if arr.dtype == bool or arr.dtype == np.bool_:
                ax.imshow(arr.astype(np.uint8), cmap='Oranges')
            else:
                vis = arr.astype(float).copy()
                vis[~np.isfinite(vis)] = np.nan
                finite_vals = vis[np.isfinite(vis)]
                if len(finite_vals) > 0:
                    p_low = np.nanpercentile(finite_vals, 5)
                    p_high = np.nanpercentile(finite_vals, 99)
                else:
                    p_low, p_high = 0, 1
                im = ax.imshow(np.clip(vis, p_low, p_high), cmap=cmap)
                plt.colorbar(im, ax=ax)
            ax.set_title(title)
            plt.savefig(os.path.join(output_dir, fname), dpi=100, bbox_inches='tight')
            plt.close()

        _save_img(divide_score, 'Divide Score', 'debug_rda_divide_score.png')
        _save_img(center_band, 'Center Band (local max)', 'debug_rda_center_band.png')
        _save_img(mask, 'Final Mask', 'debug_rda_mask.png')
        print("[debug] saved divide_axis debug (3 images)")

    def compute_ridge_guidance_maps(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString]
    ):
        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        if valley_lines:
            valley_dist = self.compute_line_distance_grid(
                valley_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )
        else:
            valley_dist = np.full(dtm.shape, np.inf, dtype=np.float32)
            valley_dist[support_mask == 0] = np.inf

        ro_cfg = self.config.get('ridge_openness', {})
        radii_m = ro_cfg.get('radii_m', [60.0, 120.0, 200.0])
        sample_step_m = float(ro_cfg.get('sample_step_m', 6.0))
        directions = int(ro_cfg.get('directions', 8))

        openness_maps = []
        for radius_m in radii_m:
            op = self.compute_positive_openness(
                dtm,
                support_mask,
                resolution,
                radius_m=float(radius_m),
                sample_step_m=sample_step_m,
                directions=directions
            )
            openness_maps.append(self.robust_normalize(op, valid))

        openness_score = np.nanmean(np.stack(openness_maps, axis=0), axis=0)

        tpi_radius = max(float(r) for r in radii_m)
        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, tpi_radius)

        norm_tpi = self.robust_normalize(tpi, valid)

        relief_to_valley = self.compute_relief_to_valley(
            dtm,
            support_mask,
            transform,
            valley_lines
        )
        relief_score = self.robust_normalize(relief_to_valley, valid)

        valley_dist_valid = valley_dist.copy().astype(np.float32)
        valley_dist_valid[~valid] = np.nan
        finite_dist = valley_dist_valid[np.isfinite(valley_dist_valid)]
        if len(finite_dist) > 0:
            dist_clip = np.nanpercentile(finite_dist, 95)
            valley_dist_clip = np.clip(valley_dist_valid, 0, dist_clip)
            dist_score = self.robust_normalize(valley_dist_clip, valid)
        else:
            dist_score = np.zeros_like(dtm, dtype=np.float32)

        ridge_score = (
            0.35 * norm_tpi
            + 0.25 * openness_score
            + 0.25 * relief_score
            + 0.15 * dist_score
        )
        ridge_score[~valid] = np.nan

        return ridge_score, tpi, valley_dist

    def collect_line_endpoints_for_gap_connect(self, lines: List[LineString]):
        endpoints = []

        for line_idx, line in enumerate(lines):
            if line.is_empty:
                continue

            coords = np.array(line.coords)
            if len(coords) < 2:
                continue

            endpoints.append({
                "line_idx": line_idx,
                "is_start": True,
                "point": coords[0],
                "direction": self.line_endpoint_direction(coords, at_start=True)
            })

            endpoints.append({
                "line_idx": line_idx,
                "is_start": False,
                "point": coords[-1],
                "direction": self.line_endpoint_direction(coords, at_start=False)
            })

        return endpoints

    def connect_ridge_gaps_by_cost_path(
        self,
        lines: List[LineString],
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        ridge_score: np.ndarray,
        tpi: np.ndarray,
        valley_dist: np.ndarray,
        cfg: dict,
        profile_score: Optional[np.ndarray] = None,
        edge_dist: Optional[np.ndarray] = None,
        corridor_score: Optional[np.ndarray] = None
    ) -> List[LineString]:

        if not lines or len(lines) < 2:
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = dtm.shape
        valid = (support_mask > 0) & np.isfinite(dtm)

        min_gap_m = float(cfg.get('min_gap_m', 8.0))
        max_gap_m = float(cfg.get('max_gap_m', 120.0))
        max_angle_deg = float(cfg.get('max_angle_deg', 80.0))
        search_margin_m = float(cfg.get('search_margin_m', 40.0))

        min_mean_score = float(cfg.get('min_mean_score', 0.42))
        min_min_score = float(cfg.get('min_min_score', 0.18))
        max_path_factor = float(cfg.get('max_path_factor', 1.8))
        max_near_valley_ratio = float(cfg.get('max_near_valley_ratio', 0.25))
        min_valley_distance_m = float(cfg.get('min_valley_distance_m', 6.0))
        min_edge_distance_m = float(cfg.get('min_edge_distance_m', 0.0))

        ridge_w = float(cfg.get('ridge_score_weight', 0.60))
        profile_w = float(cfg.get('profile_score_weight', 0.0))
        tpi_w = float(cfg.get('tpi_weight', 0.25))
        dist_w = float(cfg.get('valley_distance_weight', 0.15))

        max_connections = int(cfg.get('max_connections_per_iter', 80))
        min_connector_length = float(cfg.get('min_connector_length_m', 8.0))
        max_connector_length = float(cfg.get('max_connector_length_m', 180.0))

        endpoints = self.collect_line_endpoints_for_gap_connect(lines)
        if len(endpoints) < 2:
            return []

        endpoint_xy = np.array([ep["point"] for ep in endpoints], dtype=float)
        tree = cKDTree(endpoint_xy)

        tpi_score = self.robust_normalize(tpi, valid)

        valley_dist_clip = np.clip(valley_dist, 0, 120.0)
        dist_score = self.robust_normalize(valley_dist_clip, valid)

        if corridor_score is not None:
            guide_score = corridor_score.copy().astype(np.float32)
            guide_score[~np.isfinite(guide_score)] = 0.0
        else:
            score_norm = ridge_score.copy().astype(np.float32)
            score_norm[~np.isfinite(score_norm)] = 0.0

            if profile_score is not None:
                profile_norm = profile_score.copy().astype(np.float32)
                profile_norm[~np.isfinite(profile_norm)] = 0.0
            else:
                profile_norm = np.zeros_like(score_norm, dtype=np.float32)

            tpi_score[~np.isfinite(tpi_score)] = 0.0
            dist_score[~np.isfinite(dist_score)] = 0.0

            guide_score = (
                ridge_w * score_norm
                + profile_w * profile_norm
                + tpi_w * tpi_score
                + dist_w * dist_score
            )
        guide_score[~valid] = 0.0

        connector_candidates = []

        used_pairs = set()

        for i, ep1 in enumerate(endpoints):
            nearby = tree.query_ball_point(ep1["point"], r=max_gap_m)

            for j in nearby:
                if j <= i:
                    continue

                ep2 = endpoints[j]

                if ep1["line_idx"] == ep2["line_idx"]:
                    continue

                pair_key = tuple(sorted((i, j)))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)

                p1 = ep1["point"]
                p2 = ep2["point"]
                gap_dist = float(np.linalg.norm(p2 - p1))

                if gap_dist < min_gap_m or gap_dist > max_gap_m:
                    continue

                d1 = ep1["direction"]
                d2 = ep2["direction"]
                if d1 is None or d2 is None:
                    continue

                connect_vec = p2 - p1
                connect_norm = np.linalg.norm(connect_vec)
                if connect_norm < 1e-6:
                    continue

                connect_dir = connect_vec / connect_norm

                angle1 = self.angle_between_vectors(d1, connect_dir)
                angle2 = self.angle_between_vectors(d2, -connect_dir)

                if angle1 > max_angle_deg or angle2 > max_angle_deg:
                    continue

                c1 = int((p1[0] - transform.c) / transform.a)
                r1 = int((p1[1] - transform.f) / transform.e)
                c2 = int((p2[0] - transform.c) / transform.a)
                r2 = int((p2[1] - transform.f) / transform.e)

                if not (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols):
                    continue

                margin_cells = max(5, int(round(search_margin_m / resolution)))
                rmin = max(0, min(r1, r2) - margin_cells)
                rmax = min(rows, max(r1, r2) + margin_cells + 1)
                cmin = max(0, min(c1, c2) - margin_cells)
                cmax = min(cols, max(c1, c2) + margin_cells + 1)

                if rmax <= rmin + 2 or cmax <= cmin + 2:
                    continue

                sub_valid = valid[rmin:rmax, cmin:cmax]
                sub_guide = guide_score[rmin:rmax, cmin:cmax]
                sub_valley_dist = valley_dist[rmin:rmax, cmin:cmax]
                sub_edge_dist = edge_dist[rmin:rmax, cmin:cmax] if edge_dist is not None else None

                cost = 1.0 - sub_guide
                cost = cost.astype(np.float32)

                cost[~sub_valid] = 9999.0
                cost[sub_valley_dist < min_valley_distance_m] += 5.0
                if sub_edge_dist is not None and min_edge_distance_m > 0:
                    cost[sub_edge_dist < min_edge_distance_m] += 5.0

                start = (r1 - rmin, c1 - cmin)
                end = (r2 - rmin, c2 - cmin)

                try:
                    path, path_cost = route_through_array(
                        cost,
                        start,
                        end,
                        fully_connected=True,
                        geometric=True
                    )
                except Exception:
                    continue

                if not path or len(path) < 2:
                    continue

                path = np.array(path, dtype=int)
                rr = path[:, 0] + rmin
                cc = path[:, 1] + cmin

                path_scores = guide_score[rr, cc]
                path_vdist = valley_dist[rr, cc]
                path_edist = edge_dist[rr, cc] if edge_dist is not None else None

                finite_scores = path_scores[np.isfinite(path_scores)]
                if len(finite_scores) == 0:
                    continue

                mean_score = float(np.mean(finite_scores))
                min_score = float(np.min(finite_scores))

                near_valley_ratio = float(np.mean(path_vdist < min_valley_distance_m))
                near_edge_ratio = 0.0
                if path_edist is not None and min_edge_distance_m > 0:
                    near_edge_ratio = float(np.mean(path_edist < min_edge_distance_m))

                path_xy = []
                for r, c in zip(rr, cc):
                    x = transform.c + (c + 0.5) * transform.a
                    y = transform.f + (r + 0.5) * transform.e
                    path_xy.append([x, y])

                path_xy = np.array(path_xy, dtype=float)
                seg_len = np.sqrt(np.sum(np.diff(path_xy, axis=0) ** 2, axis=1))
                path_len = float(np.sum(seg_len))

                if path_len < min_connector_length or path_len > max_connector_length:
                    continue

                if path_len / max(gap_dist, 1e-6) > max_path_factor:
                    continue

                if mean_score < min_mean_score:
                    continue

                if min_score < min_min_score:
                    continue

                if near_valley_ratio > max_near_valley_ratio:
                    continue

                if near_edge_ratio > 0.05:
                    continue

                path_xy[0] = p1
                path_xy[-1] = p2
                connector = LineString(path_xy)
                if connector.is_empty:
                    continue

                final_score = mean_score * gap_dist / max(path_len, 1e-6)

                connector_candidates.append((connector, final_score, ep1["line_idx"], ep2["line_idx"]))

        if not connector_candidates:
            print("[ridge_gap_connect] connectors=0")
            return []

        connector_candidates.sort(key=lambda x: x[1], reverse=True)

        selected = []
        connected_lines = set()

        for connector, score, line_i, line_j in connector_candidates:
            if len(selected) >= max_connections:
                break

            pair = tuple(sorted((line_i, line_j)))
            if pair in connected_lines:
                continue

            selected.append(connector)
            connected_lines.add(pair)

        print(f"[ridge_gap_connect] candidates={len(connector_candidates)}, selected={len(selected)}")

        return selected


    def extract_divide_axis_ridge_lines(
        self, dtm: np.ndarray, support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        existing_ridge_lines: List[LineString]
    ) -> List[LineString]:
        cfg = self.config.get('ridge_divide_axis', {})
        if not cfg.get('enabled', False):
            return []

        if not valley_lines:
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        # 1. Compute distance to valley lines
        valley_dist = self.compute_line_distance_grid(
            valley_lines, dtm.shape, transform, support_mask, resolution
        )
        valley_dist_valid = valley_dist.copy().astype(np.float32)
        valley_dist_valid[~valid] = np.nan

        # Clip extreme values to avoid boundary domination
        finite_dist = valley_dist_valid[np.isfinite(valley_dist_valid)]
        if len(finite_dist) == 0:
            return []
        max_dist_clip = np.nanpercentile(finite_dist, 95)
        valley_dist_clip = np.clip(valley_dist_valid, 0, max_dist_clip)

        dist_score = self.robust_normalize(valley_dist_clip, valid)

        # 2. Large-scale TPI
        tpi_radius = float(cfg.get('tpi_radius_m', 120.0))
        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, tpi_radius)
        tpi_score = self.robust_normalize(tpi, valid)

        # 3. Curvature
        curvature = self.compute_curvature_score(dtm, support_mask, resolution)
        curv_score = self.robust_normalize(curvature, valid)

        # 4. Slope
        slope = self.compute_slope(dtm, resolution)

        # 5. Combined score
        distance_w = float(cfg.get('distance_weight', 0.40))
        tpi_w = float(cfg.get('tpi_weight', 0.45))
        curv_w = float(cfg.get('curvature_weight', 0.05))
        openness_w = float(cfg.get('openness_weight', 0.10))
        relief_w = float(cfg.get('relief_weight', 0.25))

        if openness_w > 0:
            openness_radius = float(cfg.get('openness_radius_m', cfg.get('tpi_radius_m', 160.0)))
            openness = self.compute_positive_openness(
                dtm,
                support_mask,
                resolution,
                radius_m=openness_radius,
                sample_step_m=float(cfg.get('openness_sample_step_m', 8.0)),
                directions=8
            )
            openness_score = self.robust_normalize(openness, valid)
        else:
            openness_score = np.zeros_like(dist_score, dtype=np.float32)

        relief_to_valley = self.compute_relief_to_valley(
            dtm,
            support_mask,
            transform,
            valley_lines
        )
        relief_score = self.robust_normalize(relief_to_valley, valid)

        divide_score = (
            distance_w * dist_score
            + tpi_w * tpi_score
            + curv_w * curv_score
            + openness_w * openness_score
            + relief_w * relief_score
        )
        divide_score[~valid] = np.nan

        # 6. Thresholding
        valid_score = divide_score[valid & np.isfinite(divide_score)]
        if len(valid_score) == 0:
            print("[!] ridge_divide_axis: no valid score")
            return []

        score_threshold = np.percentile(valid_score, float(cfg.get('distance_percentile', 45.0)))

        valid_tpi = tpi[valid & np.isfinite(tpi)]
        if len(valid_tpi) > 0:
            tpi_threshold = np.percentile(valid_tpi, float(cfg.get('min_tpi_percentile', 35.0)))
        else:
            tpi_threshold = 0.0

        min_dist = float(cfg.get('min_distance_to_valley', 12.0))
        max_slope_deg = float(cfg.get('max_slope_deg', 60.0))

        # 6b. Local max center band: narrow wide ridges before skeletonize
        local_win_m = float(cfg.get('local_max_window_m', 40.0))
        local_win_cells = max(3, int(round(local_win_m / resolution)))
        if local_win_cells % 2 == 0:
            local_win_cells += 1

        score_for_max = np.where(
            valid & np.isfinite(divide_score),
            divide_score,
            -9999.0
        )

        local_max = ndimage.maximum_filter(
            score_for_max,
            size=local_win_cells,
            mode='nearest'
        )

        tol = float(cfg.get('local_max_tolerance', 0.04))
        center_band = score_for_max >= (local_max - tol)
        center_band = center_band & valid

        mask = (
            valid
            & center_band
            & np.isfinite(divide_score)
            & (divide_score >= score_threshold)
            & np.isfinite(tpi)
            & (tpi >= tpi_threshold)
            & np.isfinite(slope)
            & (slope <= max_slope_deg)
            & np.isfinite(valley_dist)
            & (valley_dist >= min_dist)
        )

        if self.config.get('output', {}).get('save_broad_debug', False):
            self._save_divide_axis_debug(divide_score, center_band, mask, output_dir=self.output_dir)

        # 7. Morphological closing
        closing_disk = int(cfg.get('closing_disk', 3))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask = mask & valid
        mask = morphology.remove_small_objects(mask, min_size=int(cfg.get('min_area_cells', 80)))

        # 8. Skeleton to lines
        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            'ridge_divide_axis',
            min_length=float(cfg.get('min_line_length', 70.0))
        )

        if not lines:
            print("[ridge_divide_axis] lines=0")
            return []

        # 9. Profile validation and scoring
        profile_hw = float(cfg.get('profile_half_width_m', 45.0))
        er_min = float(cfg.get('ridge_extreme_ratio_min', 0.10))
        relief_min = float(cfg.get('ridge_relief_min', 0.2))

        scored = []
        rows_d, cols_d = divide_score.shape

        for line in lines:
            if line.is_empty:
                continue

            er, relief, vc = self.evaluate_line_extremeness(
                line, dtm, transform, resolution,
                mode='ridge',
                profile_half_width_m=profile_hw,
                n_line_samples=30,
                n_profile_samples=21
            )

            if er < er_min or relief < relief_min:
                continue

            score_vals = []
            n_samples = 30
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows_d and 0 <= c < cols_d:
                    s = divide_score[r, c]
                    if np.isfinite(s):
                        score_vals.append(s)

            mean_score = np.mean(score_vals) if score_vals else 0.0
            final_score = line.length * mean_score * (0.5 + er) * (1.0 + min(relief, 10.0) / 10.0)
            scored.append((line, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_top_n = int(cfg.get('keep_top_n', 100))
        result = [x[0] for x in scored[:keep_top_n]]

        # 10. Optional filter against existing ridges
        filter_existing = bool(cfg.get("filter_existing_ridge", False))

        if filter_existing and existing_ridge_lines and result:
            min_dist_existing = float(cfg.get("min_distance_from_existing", 18.0))
            near_ratio = float(cfg.get("near_ratio_threshold", 0.85))
            result = self.filter_supplement_lines(
                result,
                existing_ridge_lines,
                min_dist_existing,
                near_ratio,
                0,
                0
            )

        print(f"[ridge_divide_axis] score_th={score_threshold:.3f}, tpi_th={tpi_threshold:.3f}, mask_cells={int(np.sum(mask))}, lines={len(result)}")
        return result

    def extract_valley_divide_ridge_lines(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        edge_dist: Optional[np.ndarray] = None
    ) -> List[LineString]:
        """
        基于山谷线反推山脊线：
        山脊通常位于两侧山谷之间，是 valley distance 场的局部最大线。
        """
        cfg = self.config.get("valley_divide_ridge", {})
        if not cfg.get("enabled", False):
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        if not valley_lines:
            print("[valley_divide_ridge] no valley lines, skipped")
            return []

        valley_mask = self.rasterize_lines(
            valley_lines,
            dtm.shape,
            transform
        ).astype(bool)

        if np.sum(valley_mask) == 0:
            print("[valley_divide_ridge] empty valley mask, skipped")
            return []

        valley_dist = ndimage.distance_transform_edt(
            ~valley_mask,
            sampling=resolution
        ).astype(np.float32)

        valley_dist[~valid] = np.nan

        local_win_m = float(cfg.get("local_max_window_m", 90.0))
        local_win_cells = max(3, int(round(local_win_m / resolution)))
        if local_win_cells % 2 == 0:
            local_win_cells += 1

        dist_for_max = np.where(
            valid & np.isfinite(valley_dist),
            valley_dist,
            -9999.0
        )

        local_max = ndimage.maximum_filter(
            dist_for_max,
            size=local_win_cells,
            mode="nearest"
        )

        tol_m = float(cfg.get("local_max_tolerance_m", 10.0))
        center_band = dist_for_max >= (local_max - tol_m)

        tpi_radius = float(cfg.get("tpi_radius_m", 180.0))
        tpi = self.compute_tpi_raw(
            dtm,
            support_mask,
            resolution,
            tpi_radius
        )

        valid_tpi = tpi[valid & np.isfinite(tpi)]
        if valid_tpi.size == 0:
            print("[valley_divide_ridge] no valid TPI, skipped")
            return []

        tpi_th = np.percentile(
            valid_tpi,
            float(cfg.get("min_tpi_percentile", 32.0))
        )

        relief = self.compute_relief_to_valley(
            dtm,
            support_mask,
            transform,
            valley_lines
        )

        min_dist = float(cfg.get("min_distance_to_valley_m", 18.0))
        min_relief = float(cfg.get("min_relief_to_valley_m", 1.5))

        mask = (
            valid
            & center_band
            & np.isfinite(valley_dist)
            & (valley_dist >= min_dist)
            & np.isfinite(tpi)
            & (tpi >= tpi_th)
            & np.isfinite(relief)
            & (relief >= min_relief)
        )

        if edge_dist is not None:
            min_edge = float(cfg.get("min_edge_distance_m", 50.0))
            mask = mask & (edge_dist >= min_edge)

        closing_disk = int(cfg.get("closing_disk", 2))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask = mask & valid

        min_area = int(cfg.get("min_area_cells", 40))
        if min_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_area)

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            "valley_divide_ridge",
            min_length=float(cfg.get("min_line_length", 90.0))
        )

        if not lines:
            print("[valley_divide_ridge] lines=0")
            return []

        scored = []
        rows, cols = valley_dist.shape

        for line in lines:
            vals_dist = []
            vals_relief = []

            n_samples = 30
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(valley_dist[r, c]):
                        vals_dist.append(valley_dist[r, c])
                    if np.isfinite(relief[r, c]):
                        vals_relief.append(relief[r, c])

            mean_dist = float(np.mean(vals_dist)) if vals_dist else 0.0
            mean_relief = float(np.mean(vals_relief)) if vals_relief else 0.0

            score = line.length * (0.6 * mean_dist + 0.4 * mean_relief)
            scored.append((line, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        keep_top_n = int(cfg.get("keep_top_n", 120))
        result = [x[0] for x in scored[:keep_top_n]]

        print(f"[valley_divide_ridge] lines={len(result)}")
        return result

    def select_major_valley_lines_for_watershed(
        self,
        valley_lines: List[LineString],
        accumulation: np.ndarray,
        transform: Affine,
        min_length_m: float = 120.0,
        keep_top_n: int = 80
    ) -> List[LineString]:
        """
        Select the main valley lines used as basin outlets for watershed divides.
        Using every small valley would create many fragmented divide ridges.
        """
        if not valley_lines:
            return []

        rows, cols = accumulation.shape
        scored = []

        for line in valley_lines:
            if line is None or line.is_empty or line.length < min_length_m:
                continue

            acc_vals = []
            n_samples = 40

            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    v = accumulation[r, c]
                    if np.isfinite(v):
                        acc_vals.append(v)

            mean_acc = float(np.mean(acc_vals)) if acc_vals else 0.0
            score = line.length * np.log1p(mean_acc)
            scored.append((line, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:keep_top_n]]

    def assign_basin_labels_to_valleys(
        self,
        flow_to_r: np.ndarray,
        flow_to_c: np.ndarray,
        support_mask: np.ndarray,
        valley_label_grid: np.ndarray
    ) -> np.ndarray:
        """
        Assign each supported cell to the downstream main-valley label it drains into.
        valley_label_grid already stores one independent label per main valley line.
        """
        rows, cols = support_mask.shape

        basin = np.zeros((rows, cols), dtype=np.int32)
        basin[valley_label_grid > 0] = valley_label_grid[valley_label_grid > 0]

        visiting = np.zeros((rows, cols), dtype=np.uint8)

        def resolve_cell(r: int, c: int) -> int:
            path = []

            while True:
                if r < 0 or c < 0 or r >= rows or c >= cols:
                    label = 0
                    break

                if support_mask[r, c] == 0:
                    label = 0
                    break

                if basin[r, c] > 0:
                    label = int(basin[r, c])
                    break

                if visiting[r, c] == 1:
                    label = 0
                    break

                visiting[r, c] = 1
                path.append((r, c))

                nr = flow_to_r[r, c]
                nc = flow_to_c[r, c]

                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    label = 0
                    break

                if nr == r and nc == c:
                    label = 0
                    break

                r, c = nr, nc

            for pr, pc in path:
                basin[pr, pc] = label
                visiting[pr, pc] = 0

            return label

        for r in range(rows):
            for c in range(cols):
                if support_mask[r, c] > 0 and basin[r, c] == 0:
                    resolve_cell(r, c)

        return basin

    def rasterize_labeled_valley_lines(
        self,
        lines: List[LineString],
        shape: Tuple[int, int],
        transform: Affine
    ) -> np.ndarray:
        """
        Rasterize each main valley line into a separate label.
        This avoids merging downstream-connected valley networks into one outlet.
        """
        label_grid = np.zeros(shape, dtype=np.int32)

        for idx, line in enumerate(lines, start=1):
            if line is None or line.is_empty:
                continue

            mask = rio_features.rasterize(
                shapes=[(line, idx)],
                out_shape=shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.int32
            )

            label_grid[(label_grid == 0) & (mask > 0)] = idx

        return label_grid

    def fill_unassigned_basin_labels(
        self,
        basin: np.ndarray,
        valid: np.ndarray,
        max_fill_distance_cells: int = 80
    ) -> np.ndarray:
        """
        Fill valid basin=0 cells from the nearest non-zero basin label.
        This is only used inside watershed divide ridge extraction.
        """
        known = (basin > 0) & valid
        unknown = (basin == 0) & valid

        if np.sum(known) == 0:
            return basin

        dist, inds = ndimage.distance_transform_edt(
            ~known,
            return_indices=True
        )

        rr = inds[0]
        cc = inds[1]

        filled = basin.copy()
        fill_mask = unknown & (dist <= max_fill_distance_cells)
        filled[fill_mask] = basin[rr[fill_mask], cc[fill_mask]]

        return filled

    def compute_terrain_active_mask(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        resolution: float,
        cfg: Optional[dict] = None
    ) -> np.ndarray:
        """Build a broad terrain-relief mask used only by ridge top extraction."""
        cfg = cfg or self.config.get("terrain_active", {})
        valid = (support_mask > 0) & np.isfinite(dtm)

        if not cfg.get("enabled", False):
            return valid

        radii = cfg.get("relief_radius_m", [80.0, 160.0])
        thresholds = cfg.get("min_relief_m", [1.0, 1.5])

        if not isinstance(radii, (list, tuple)):
            radii = [radii]
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]
        if len(thresholds) < len(radii):
            thresholds = list(thresholds) + [thresholds[-1]] * (len(radii) - len(thresholds))

        dtm_for_max = np.where(valid, dtm, -np.inf)
        dtm_for_min = np.where(valid, dtm, np.inf)
        active = np.zeros_like(valid, dtype=bool)

        for radius_m, min_relief_m in zip(radii, thresholds):
            radius_cells = max(1, int(round(float(radius_m) / resolution)))
            size = 2 * radius_cells + 1
            local_max = ndimage.maximum_filter(dtm_for_max, size=size, mode="nearest")
            local_min = ndimage.minimum_filter(dtm_for_min, size=size, mode="nearest")
            local_relief = local_max - local_min
            active |= valid & np.isfinite(local_relief) & (local_relief >= float(min_relief_m))

        return active & valid

    def select_important_valley_lines(
        self,
        valley_lines: List[LineString],
        dtm: np.ndarray,
        accumulation: np.ndarray,
        transform: Affine,
        support_mask: np.ndarray,
        cfg: dict,
        mode: str = "valley"
    ) -> List[LineString]:
        """
        Keep major valley lines for ridge guidance and optionally final output.
        Small shallow gullies are intentionally filtered out.
        """
        if not valley_lines:
            return []

        if not cfg.get("enabled", False):
            return list(valley_lines)

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = accumulation.shape
        min_length = float(cfg.get("min_line_length_m", 120.0))
        profile_hw = float(cfg.get("profile_half_width_m", 45.0))
        min_extreme = float(cfg.get("min_valley_extreme_ratio", 0.45))
        min_relief = float(cfg.get("min_valley_relief_m", 1.2))

        candidates = []
        for line in valley_lines:
            if line is None or line.is_empty or line.length < min_length:
                continue

            acc_vals = []
            n_samples = int(cfg.get("sample_points", 40))
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows and 0 <= c < cols:
                    if support_mask[r, c] > 0 and np.isfinite(accumulation[r, c]):
                        acc_vals.append(accumulation[r, c])

            mean_acc = float(np.mean(acc_vals)) if acc_vals else 0.0
            extreme_ratio, mean_relief, valid_count = self.evaluate_line_extremeness(
                line,
                dtm,
                transform,
                resolution,
                mode=mode,
                profile_half_width_m=profile_hw,
                n_line_samples=30,
                n_profile_samples=21
            )

            candidates.append({
                "line": line,
                "length": float(line.length),
                "mean_acc": mean_acc,
                "extreme_ratio": float(extreme_ratio),
                "mean_relief": float(mean_relief),
                "valid_count": int(valid_count)
            })

        if not candidates:
            print("[major_valley_filter] selected=0")
            return []

        acc_values = np.array([c["mean_acc"] for c in candidates], dtype=float)
        min_acc_percentile = float(cfg.get("min_mean_acc_percentile", 60.0))
        acc_threshold = float(np.percentile(acc_values, min_acc_percentile)) if acc_values.size else 0.0

        filtered = [
            c for c in candidates
            if c["mean_acc"] >= acc_threshold
            and c["extreme_ratio"] >= min_extreme
            and c["mean_relief"] >= min_relief
            and c["valid_count"] > 0
        ]

        if not filtered:
            print(
                f"[major_valley_filter] selected=0, candidates={len(candidates)}, "
                f"acc_th={acc_threshold:.2f}"
            )
            return []

        def safe_norm(values):
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                return arr
            p10 = np.percentile(arr, 10)
            p90 = np.percentile(arr, 90)
            spread = p90 - p10
            if spread < 1e-6:
                return np.zeros_like(arr)
            return np.clip((arr - p10) / spread, 0.0, 1.0)

        norm_length = safe_norm([c["length"] for c in filtered])
        norm_acc = safe_norm([c["mean_acc"] for c in filtered])
        norm_relief = safe_norm([c["mean_relief"] for c in filtered])

        scored = []
        for i, item in enumerate(filtered):
            score = (
                0.35 * norm_length[i]
                + 0.30 * norm_acc[i]
                + 0.20 * item["extreme_ratio"]
                + 0.15 * norm_relief[i]
            )
            scored.append((item["line"], float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_top_n = int(cfg.get("keep_top_n", 80))
        selected = [line for line, _ in scored[:keep_top_n]]

        print(
            f"[major_valley_filter] all={len(valley_lines)}, candidates={len(candidates)}, "
            f"selected={len(selected)}, acc_th={acc_threshold:.2f}"
        )
        return selected

    def compute_profile_ridge_score(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        resolution: float,
        cfg: Optional[dict] = None
    ) -> np.ndarray:
        """
        Score ridge-top pixels by cross-profile height drops.
        A high score means the center is higher than samples on both sides.
        """
        cfg = cfg or self.config.get("profile_ridge", {})
        valid = (support_mask > 0) & np.isfinite(dtm)

        distances_m = cfg.get("sample_distances_m", [12.0, 24.0, 48.0, 96.0])
        if not isinstance(distances_m, (list, tuple)):
            distances_m = [distances_m]

        n_dirs = int(cfg.get("directions", 8))
        n_dirs = max(4, n_dirs)
        shoulder_factor = float(cfg.get("shoulder_factor", 0.65))

        direction_scores = []
        for i in range(n_dirs):
            theta = np.pi * i / n_dirs
            dist_scores = []

            for dist_m in distances_m:
                dist_cells = max(1, int(round(float(dist_m) / resolution)))
                dc = int(round(np.cos(theta) * dist_cells))
                dr = int(round(np.sin(theta) * dist_cells))

                if dr == 0 and dc == 0:
                    continue

                left_z = self.sample_offset(dtm, dr, dc, fill=np.nan)
                right_z = self.sample_offset(dtm, -dr, -dc, fill=np.nan)

                left_valid = self.sample_offset(
                    valid.astype(np.float32), dr, dc, fill=0.0
                ) > 0.5
                right_valid = self.sample_offset(
                    valid.astype(np.float32), -dr, -dc, fill=0.0
                ) > 0.5

                drop_left = dtm - left_z
                drop_right = dtm - right_z

                bilateral = np.minimum(drop_left, drop_right)
                shoulder = 0.5 * np.maximum(drop_left, drop_right) + 0.5 * bilateral
                direction_score = np.maximum(bilateral, shoulder_factor * shoulder)
                direction_score = np.where(
                    valid & left_valid & right_valid & np.isfinite(direction_score),
                    np.maximum(direction_score, 0.0),
                    np.nan
                ).astype(np.float32)
                dist_scores.append(direction_score)

            if dist_scores:
                dist_stack = np.stack(dist_scores, axis=0)
                dist_stack = np.where(np.isfinite(dist_stack), dist_stack, -np.inf)
                direction_score = np.max(dist_stack, axis=0).astype(np.float32)
                direction_score[~np.isfinite(direction_score)] = np.nan
                direction_scores.append(direction_score)

        raw = np.zeros_like(dtm, dtype=np.float32)
        if direction_scores:
            direction_stack = np.stack(direction_scores, axis=0)
            finite_stack = np.where(np.isfinite(direction_stack), direction_stack, -np.inf)
            sorted_scores = np.sort(finite_stack, axis=0)
            if sorted_scores.shape[0] >= 2:
                top = sorted_scores[-2:, :, :]
                top = np.where(np.isfinite(top), top, np.nan)
                with np.errstate(invalid="ignore"):
                    raw = np.nanmean(top, axis=0).astype(np.float32)
            else:
                raw = sorted_scores[-1, :, :].astype(np.float32)
                raw[~np.isfinite(raw)] = np.nan

        raw[~valid] = np.nan
        return self.robust_normalize(raw, valid)

    def extract_openness_top_ridge_lines(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        edge_dist: Optional[np.ndarray] = None,
        terrain_active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[LineString], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract primary ridge-top lines with positive openness.
        This is independent from the legacy ridge_openness module.
        """
        cfg = self.config.get("ridge_openness_top", {})
        if not cfg.get("enabled", False):
            return [], None, None

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)
        active = valid.copy()
        if terrain_active_mask is not None:
            active &= terrain_active_mask.astype(bool)

        if np.sum(active) == 0:
            print("[ridge_openness_top] no active terrain pixels")
            ridge_score = np.full(dtm.shape, np.nan, dtype=np.float32)
            return [], ridge_score, None

        radii_m = cfg.get("radii_m", [60.0, 120.0, 200.0])
        sample_step_m = float(cfg.get("sample_step_m", 8.0))
        directions = int(cfg.get("directions", 8))

        openness_maps = []
        for radius_m in radii_m:
            op = self.compute_positive_openness(
                dtm,
                support_mask,
                resolution,
                radius_m=float(radius_m),
                sample_step_m=sample_step_m,
                directions=directions
            )
            openness_maps.append(self.robust_normalize(op, active))

        with np.errstate(invalid="ignore"):
            openness_score = np.nanmean(np.stack(openness_maps, axis=0), axis=0).astype(np.float32)

        profile_score = self.compute_profile_ridge_score(
            dtm,
            support_mask,
            resolution,
            self.config.get("profile_ridge", {})
        )

        tpi_radius_m = max(float(r) for r in radii_m) if radii_m else 180.0
        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, tpi_radius_m)
        tpi_score = self.robust_normalize(tpi, active)

        relief_radius_m = float(cfg.get("relief_radius_m", cfg.get("local_max_window_m", 80.0)))
        relief_radius_cells = max(1, int(round(relief_radius_m / resolution)))
        relief_size = 2 * relief_radius_cells + 1
        dtm_for_max = np.where(valid, dtm, -np.inf)
        dtm_for_min = np.where(valid, dtm, np.inf)
        local_max_z = ndimage.maximum_filter(dtm_for_max, size=relief_size, mode="nearest")
        local_min_z = ndimage.minimum_filter(dtm_for_min, size=relief_size, mode="nearest")
        local_relief = (local_max_z - local_min_z).astype(np.float32)
        local_relief[~valid] = np.nan
        relief_score = self.robust_normalize(local_relief, active)

        valley_dist = None
        valley_support = np.zeros(dtm.shape, dtype=np.float32)
        if valley_lines:
            valley_dist = self.compute_line_distance_grid(
                valley_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )
            min_valley_dist = float(cfg.get("min_distance_to_valley_m", 8.0))
            max_valley_dist = float(cfg.get("max_distance_to_valley_m", 220.0))
            if max_valley_dist > min_valley_dist:
                valley_support = np.clip(
                    (valley_dist - min_valley_dist) / (max_valley_dist - min_valley_dist),
                    0.0,
                    1.0
                ).astype(np.float32)
            valley_support[~valid] = np.nan

        profile_w = float(cfg.get("profile_weight", 0.40))
        openness_w = float(cfg.get("openness_weight", 0.30))
        tpi_w = float(cfg.get("tpi_weight", 0.18))
        relief_w = float(cfg.get("relief_weight", 0.10))
        valley_w = float(cfg.get("valley_support_weight", 0.02))

        ridge_score = (
            profile_w * np.nan_to_num(profile_score, nan=0.0)
            + openness_w * np.nan_to_num(openness_score, nan=0.0)
            + tpi_w * np.nan_to_num(tpi_score, nan=0.0)
            + relief_w * np.nan_to_num(relief_score, nan=0.0)
            + valley_w * np.nan_to_num(valley_support, nan=0.0)
        ).astype(np.float32)
        ridge_score[~active] = np.nan

        valid_score = ridge_score[active & np.isfinite(ridge_score)]
        if valid_score.size == 0:
            print("[ridge_openness_top] no valid score")
            return [], ridge_score, profile_score

        score_th = np.percentile(valid_score, float(cfg.get("score_percentile", 72.0)))

        valid_tpi = tpi[active & np.isfinite(tpi)]
        if valid_tpi.size == 0:
            print("[ridge_openness_top] no valid TPI")
            return [], ridge_score, profile_score
        tpi_th = np.percentile(valid_tpi, float(cfg.get("min_tpi_percentile", 30.0)))

        local_win_m = float(cfg.get("local_max_window_m", 80.0))
        local_win_cells = max(3, int(round(local_win_m / resolution)))
        if local_win_cells % 2 == 0:
            local_win_cells += 1
        score_for_max = np.where(active & np.isfinite(ridge_score), ridge_score, -9999.0)
        local_score_max = ndimage.maximum_filter(
            score_for_max,
            size=local_win_cells,
            mode="nearest"
        )
        center_band = score_for_max >= (
            local_score_max - float(cfg.get("local_max_tolerance", 0.06))
        )

        mask = (
            active
            & center_band
            & np.isfinite(ridge_score)
            & (ridge_score >= score_th)
            & np.isfinite(tpi)
            & (tpi >= tpi_th)
            & np.isfinite(local_relief)
            & (local_relief >= float(cfg.get("min_local_relief_m", 1.0)))
        )

        if valley_dist is not None:
            mask &= np.isfinite(valley_dist)
            mask &= valley_dist >= float(cfg.get("min_distance_to_valley_m", 8.0))
            max_dist = float(cfg.get("max_distance_to_valley_m", 220.0))
            if 0 < max_dist < 9999.0:
                mask &= valley_dist <= max_dist

        if edge_dist is not None:
            mask &= edge_dist >= float(cfg.get("min_edge_distance_m", 45.0))

        closing_disk = int(cfg.get("closing_disk", 1))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask &= active

        min_area = int(cfg.get("min_area_cells", 35))
        if min_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_area)

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            "ridge_openness_top",
            min_length=float(cfg.get("min_line_length", 100.0))
        )

        if not lines:
            print("[ridge_openness_top] lines=0")
            return [], ridge_score, profile_score

        rows, cols = dtm.shape
        scored = []
        for line in lines:
            score_vals = []
            relief_vals = []
            n_samples = 40
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(ridge_score[r, c]):
                        score_vals.append(ridge_score[r, c])
                    if np.isfinite(local_relief[r, c]):
                        relief_vals.append(local_relief[r, c])

            mean_score = float(np.mean(score_vals)) if score_vals else 0.0
            mean_relief = float(np.mean(relief_vals)) if relief_vals else 0.0
            final_score = line.length * mean_score * (1.0 + min(mean_relief, 10.0) / 20.0)
            scored.append((line, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_top_n = int(cfg.get("keep_top_n", 120))
        result = [x[0] for x in scored[:keep_top_n]]

        print(
            f"[ridge_openness_top] score_th={score_th:.3f}, tpi_th={tpi_th:.3f}, "
            f"mask_cells={int(np.sum(mask))}, lines={len(result)}"
        )
        return result, ridge_score, profile_score

    def extract_broad_crest_ridge_lines(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        existing_ridge_lines: List[LineString],
        edge_dist: Optional[np.ndarray] = None,
        terrain_active_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[LineString], Optional[np.ndarray]]:
        """
        Supplement broad, gently rounded main crest lines.
        This module is deliberately conservative and does not replace ridge_openness_top.
        """
        cfg = self.config.get("broad_crest_ridge", {})
        if not cfg.get("enabled", False):
            return [], None

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)
        active = valid.copy()
        if terrain_active_mask is not None:
            active &= terrain_active_mask.astype(bool)

        if np.sum(active) == 0:
            print("[broad_crest_ridge] no active terrain pixels")
            return [], None

        radii_m = cfg.get("radii_m", [80.0, 160.0, 240.0, 320.0])
        sample_step_m = float(cfg.get("sample_step_m", 10.0))
        directions = int(cfg.get("directions", 8))

        openness_maps = []
        for radius_m in radii_m:
            op = self.compute_positive_openness(
                dtm,
                support_mask,
                resolution,
                radius_m=float(radius_m),
                sample_step_m=sample_step_m,
                directions=directions
            )
            openness_maps.append(self.robust_normalize(op, active))

        with np.errstate(invalid="ignore"):
            openness_score = np.nanmean(np.stack(openness_maps, axis=0), axis=0).astype(np.float32)

        profile_cfg = {
            "sample_distances_m": cfg.get("profile_distances_m", [40.0, 80.0, 160.0, 240.0]),
            "directions": directions,
            "shoulder_factor": cfg.get(
                "shoulder_factor",
                self.config.get("profile_ridge", {}).get("shoulder_factor", 0.50)
            )
        }
        profile_score = self.compute_profile_ridge_score(
            dtm,
            support_mask,
            resolution,
            profile_cfg
        )

        tpi_radius_m = max(float(r) for r in radii_m) if radii_m else 240.0
        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, tpi_radius_m)
        tpi_score = self.robust_normalize(tpi, active)

        relief_radius_m = float(cfg.get("relief_radius_m", cfg.get("local_max_window_m", 180.0)))
        relief_radius_cells = max(1, int(round(relief_radius_m / resolution)))
        relief_size = 2 * relief_radius_cells + 1
        dtm_for_max = np.where(valid, dtm, -np.inf)
        dtm_for_min = np.where(valid, dtm, np.inf)
        local_max_z = ndimage.maximum_filter(dtm_for_max, size=relief_size, mode="nearest")
        local_min_z = ndimage.minimum_filter(dtm_for_min, size=relief_size, mode="nearest")
        local_relief = (local_max_z - local_min_z).astype(np.float32)
        local_relief[~valid] = np.nan
        relief_score = self.robust_normalize(local_relief, active)

        broad_score = (
            float(cfg.get("profile_weight", 0.38)) * np.nan_to_num(profile_score, nan=0.0)
            + float(cfg.get("openness_weight", 0.28)) * np.nan_to_num(openness_score, nan=0.0)
            + float(cfg.get("tpi_weight", 0.22)) * np.nan_to_num(tpi_score, nan=0.0)
            + float(cfg.get("relief_weight", 0.12)) * np.nan_to_num(relief_score, nan=0.0)
        ).astype(np.float32)
        broad_score[~active] = np.nan

        valid_score = broad_score[active & np.isfinite(broad_score)]
        if valid_score.size == 0:
            print("[broad_crest_ridge] no valid score")
            return [], broad_score

        score_th = np.percentile(valid_score, float(cfg.get("score_percentile", 72.0)))

        valid_tpi = tpi[active & np.isfinite(tpi)]
        if valid_tpi.size == 0:
            print("[broad_crest_ridge] no valid TPI")
            return [], broad_score
        tpi_th = np.percentile(valid_tpi, float(cfg.get("min_tpi_percentile", 32.0)))

        local_win_m = float(cfg.get("local_max_window_m", 180.0))
        local_win_cells = max(3, int(round(local_win_m / resolution)))
        if local_win_cells % 2 == 0:
            local_win_cells += 1
        score_for_max = np.where(active & np.isfinite(broad_score), broad_score, -9999.0)
        local_score_max = ndimage.maximum_filter(score_for_max, size=local_win_cells, mode="nearest")
        center_band = score_for_max >= (
            local_score_max - float(cfg.get("local_max_tolerance", 0.06))
        )

        mask = (
            active
            & center_band
            & np.isfinite(broad_score)
            & (broad_score >= score_th)
            & np.isfinite(tpi)
            & (tpi >= tpi_th)
            & np.isfinite(local_relief)
            & (local_relief >= float(cfg.get("min_local_relief_m", 1.0)))
        )

        if valley_lines:
            valley_dist = self.compute_line_distance_grid(
                valley_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )
            mask &= np.isfinite(valley_dist)
            mask &= valley_dist >= float(cfg.get("min_distance_to_valley_m", 6.0))

        if edge_dist is not None:
            mask &= edge_dist >= float(cfg.get("min_edge_distance_m", 35.0))

        closing_disk = int(cfg.get("closing_disk", 2))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask &= active

        min_area = int(cfg.get("min_area_cells", 80))
        if min_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_area)

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            "broad_crest_ridge",
            min_length=float(cfg.get("min_line_length", 160.0))
        )

        if not lines:
            print("[broad_crest_ridge] lines=0")
            return [], broad_score

        rows, cols = dtm.shape
        scored = []
        for line in lines:
            score_vals = []
            relief_vals = []
            n_samples = 40
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(broad_score[r, c]):
                        score_vals.append(broad_score[r, c])
                    if np.isfinite(local_relief[r, c]):
                        relief_vals.append(local_relief[r, c])

            mean_score = float(np.mean(score_vals)) if score_vals else 0.0
            mean_relief = float(np.mean(relief_vals)) if relief_vals else 0.0
            final_score = line.length * mean_score * (1.0 + min(mean_relief, 15.0) / 30.0)
            scored.append((line, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        keep_top_n = int(cfg.get("keep_top_n", 60))
        result = [line for line, _ in scored[:keep_top_n]]

        if existing_ridge_lines and result:
            result = self.filter_supplement_lines(
                result,
                existing_ridge_lines,
                float(cfg.get("min_distance_from_existing_ridge_m", 35.0)),
                float(cfg.get("near_ratio_threshold", 0.85)),
                0.0,
                0
            )

        print(
            f"[broad_crest_ridge] score_th={score_th:.3f}, tpi_th={tpi_th:.3f}, "
            f"mask_cells={int(np.sum(mask))}, lines={len(result)}"
        )
        return result, broad_score

    def compute_ridge_corridor_score(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        ridge_top_score: Optional[np.ndarray],
        profile_score: Optional[np.ndarray],
        edge_dist: Optional[np.ndarray] = None,
        terrain_active_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a controlled broad ridge corridor score for gap connection only.
        It is not directly converted to ridge lines.
        """
        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)
        if terrain_active_mask is not None:
            valid = valid & terrain_active_mask.astype(bool)

        tpi = self.compute_tpi_raw(dtm, support_mask, resolution, 180.0)
        tpi_score = self.robust_normalize(tpi, valid)

        openness = self.compute_positive_openness(
            dtm,
            support_mask,
            resolution,
            radius_m=160.0,
            sample_step_m=8.0,
            directions=8
        )
        openness_score = self.robust_normalize(openness, valid)

        relief_radius_cells = max(1, int(round(160.0 / resolution)))
        relief_size = 2 * relief_radius_cells + 1
        dtm_for_max = np.where(valid, dtm, -np.inf)
        dtm_for_min = np.where(valid, dtm, np.inf)
        local_relief = (
            ndimage.maximum_filter(dtm_for_max, size=relief_size, mode="nearest")
            - ndimage.minimum_filter(dtm_for_min, size=relief_size, mode="nearest")
        ).astype(np.float32)
        local_relief[~valid] = np.nan
        relief_score = self.robust_normalize(local_relief, valid)

        if valley_lines:
            valley_dist = self.compute_line_distance_grid(
                valley_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )
        else:
            valley_dist = np.full(dtm.shape, np.inf, dtype=np.float32)
            valley_dist[~valid] = np.nan

        valley_dist_clip = np.clip(valley_dist, 0.0, 120.0)
        valley_dist_score = self.robust_normalize(valley_dist_clip, valid)

        if ridge_top_score is None:
            ridge_top = np.zeros_like(dtm, dtype=np.float32)
        else:
            ridge_top = ridge_top_score.copy().astype(np.float32)
            ridge_top[~np.isfinite(ridge_top)] = 0.0

        if profile_score is None:
            profile = np.zeros_like(dtm, dtype=np.float32)
        else:
            profile = profile_score.copy().astype(np.float32)
            profile[~np.isfinite(profile)] = 0.0

        ridge_or_open = np.maximum(ridge_top, np.nan_to_num(openness_score, nan=0.0))

        corridor = (
            0.35 * ridge_or_open
            + 0.25 * profile
            + 0.20 * np.nan_to_num(tpi_score, nan=0.0)
            + 0.10 * np.nan_to_num(relief_score, nan=0.0)
            + 0.10 * np.nan_to_num(valley_dist_score, nan=0.0)
        ).astype(np.float32)

        corridor[~valid] = np.nan
        corridor[np.isfinite(valley_dist) & (valley_dist < 6.0)] *= 0.15
        if edge_dist is not None:
            corridor[edge_dist < 70.0] *= 0.15

        return corridor, tpi, valley_dist

    def extract_watershed_divide_ridge_lines(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        flow_to_r: np.ndarray,
        flow_to_c: np.ndarray,
        accumulation: np.ndarray,
        valley_flow_lines: List[LineString],
        valley_broad_lines: List[LineString],
        edge_dist: Optional[np.ndarray] = None
    ) -> List[LineString]:
        """
        Extract ridge lines as watershed divides between basins draining to main valleys.
        """
        cfg = self.config.get("watershed_divide_ridge", {})
        if not cfg.get("enabled", False):
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        major_valleys_for_label = self.select_major_valley_lines_for_watershed(
            valley_flow_lines,
            accumulation,
            transform,
            min_length_m=float(cfg.get("min_valley_line_length_m", 70.0)),
            keep_top_n=int(cfg.get("major_valley_keep_top_n", 150))
        )

        if not major_valleys_for_label:
            print("[watershed_divide_ridge] no major valley flow lines")
            return []

        valleys_for_measure = list(major_valleys_for_label)
        if cfg.get("use_broad_valley_for_measure", True):
            valleys_for_measure += valley_broad_lines

        valley_label_grid = self.rasterize_labeled_valley_lines(
            major_valleys_for_label,
            dtm.shape,
            transform
        )

        if np.sum(valley_label_grid > 0) == 0:
            print("[watershed_divide_ridge] empty valley_label_grid")
            return []

        measure_valley_mask = self.rasterize_lines(
            valleys_for_measure,
            dtm.shape,
            transform
        ).astype(bool)

        measure_valley_mask = morphology.binary_dilation(measure_valley_mask, morphology.disk(1))
        measure_valley_mask = morphology.binary_closing(measure_valley_mask, morphology.disk(1))
        measure_valley_mask = measure_valley_mask & valid

        basin = self.assign_basin_labels_to_valleys(
            flow_to_r,
            flow_to_c,
            support_mask,
            valley_label_grid
        )

        max_fill_m = float(cfg.get("max_unassigned_fill_distance_m", 120.0))
        max_fill_cells = max(1, int(round(max_fill_m / resolution)))
        basin = self.fill_unassigned_basin_labels(
            basin,
            valid,
            max_fill_distance_cells=max_fill_cells
        )

        min_basin_area = int(cfg.get("min_basin_area_cells", 120))
        if min_basin_area > 0:
            labels, counts = np.unique(basin[basin > 0], return_counts=True)
            small_labels = set(labels[counts < min_basin_area].tolist())
            if small_labels:
                small_mask = np.isin(basin, list(small_labels))
                basin[small_mask] = 0
                basin = self.fill_unassigned_basin_labels(
                    basin,
                    valid,
                    max_fill_distance_cells=max_fill_cells
                )

        rows, cols = basin.shape
        boundary = np.zeros_like(valid, dtype=bool)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                shifted = np.zeros_like(basin)

                if dr >= 0:
                    src_r0, src_r1 = 0, rows - dr
                    dst_r0, dst_r1 = dr, rows
                else:
                    src_r0, src_r1 = -dr, rows
                    dst_r0, dst_r1 = 0, rows + dr

                if dc >= 0:
                    src_c0, src_c1 = 0, cols - dc
                    dst_c0, dst_c1 = dc, cols
                else:
                    src_c0, src_c1 = -dc, cols
                    dst_c0, dst_c1 = 0, cols + dc

                shifted[dst_r0:dst_r1, dst_c0:dst_c1] = basin[src_r0:src_r1, src_c0:src_c1]

                boundary |= (
                    valid
                    & (basin > 0)
                    & (shifted > 0)
                    & (basin != shifted)
                )

        valley_dist, hand = self.compute_distance_and_relief_from_mask(
            dtm,
            support_mask,
            measure_valley_mask,
            resolution
        )

        tpi = self.compute_tpi_raw(
            dtm,
            support_mask,
            resolution,
            float(cfg.get("tpi_radius_m", 180.0))
        )

        valid_tpi = tpi[valid & np.isfinite(tpi)]
        if valid_tpi.size == 0:
            print("[watershed_divide_ridge] no valid TPI")
            return []

        tpi_th = np.percentile(
            valid_tpi,
            float(cfg.get("min_tpi_percentile", 23.0))
        )

        boundary_dilation_m = float(cfg.get("boundary_dilation_m", 10.0))
        boundary_dilation_cells = max(1, int(round(boundary_dilation_m / resolution)))
        boundary_band = morphology.binary_dilation(
            boundary,
            morphology.disk(boundary_dilation_cells)
        )

        mask = (
            boundary_band
            & valid
            & np.isfinite(valley_dist)
            & (valley_dist >= float(cfg.get("min_distance_to_valley_m", 12.0)))
            & np.isfinite(hand)
            & (hand >= float(cfg.get("min_hand_m", 0.8)))
            & np.isfinite(tpi)
            & (tpi >= tpi_th)
        )

        if edge_dist is not None:
            mask = mask & (
                edge_dist >= float(cfg.get("min_edge_distance_m", 45.0))
            )

        closing_disk = int(cfg.get("closing_disk", 2))
        if closing_disk > 0:
            mask = morphology.binary_closing(mask, morphology.disk(closing_disk))

        mask = mask & valid

        min_area = int(cfg.get("min_area_cells", 25))
        if min_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_area)

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            "watershed_divide_ridge",
            min_length=float(cfg.get("min_line_length", 65.0))
        )

        if not lines:
            print("[watershed_divide_ridge] lines=0")
            return []

        scored = []
        for line in lines:
            vals_hand = []
            vals_dist = []
            vals_tpi = []

            n_samples = 40
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(hand[r, c]):
                        vals_hand.append(hand[r, c])
                    if np.isfinite(valley_dist[r, c]):
                        vals_dist.append(valley_dist[r, c])
                    if np.isfinite(tpi[r, c]):
                        vals_tpi.append(tpi[r, c])

            mean_hand = float(np.mean(vals_hand)) if vals_hand else 0.0
            mean_dist = float(np.mean(vals_dist)) if vals_dist else 0.0
            mean_tpi = float(np.mean(vals_tpi)) if vals_tpi else 0.0

            score = line.length * (
                0.45 * mean_hand
                + 0.35 * mean_dist
                + 0.20 * max(mean_tpi, 0.0)
            )

            scored.append((line, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        keep_top_n = int(cfg.get("keep_top_n", 220))
        result = [x[0] for x in scored[:keep_top_n]]

        print(f"[watershed_divide_ridge] lines={len(result)}")
        return result

    def extract_valley_center_supplement_ridge_lines(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        valley_lines: List[LineString],
        existing_ridge_lines: List[LineString],
        edge_dist: Optional[np.ndarray] = None
    ) -> List[LineString]:
        """
        Supplement ridge lines along center zones between valleys.
        This fills broad or broken inter-valley ridges missed by watershed divides.
        """
        cfg = self.config.get("ridge_center_supplement", {})
        if not cfg.get("enabled", False):
            return []

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        valid = (support_mask > 0) & np.isfinite(dtm)

        if not valley_lines:
            print("[ridge_center_supplement] no valley lines")
            return []

        valley_mask = self.rasterize_lines(
            valley_lines,
            dtm.shape,
            transform
        ).astype(bool)

        if np.sum(valley_mask) == 0:
            print("[ridge_center_supplement] empty valley mask")
            return []

        valley_mask = morphology.binary_dilation(
            valley_mask,
            morphology.disk(1)
        )
        valley_mask = valley_mask & valid

        valley_dist, hand = self.compute_distance_and_relief_from_mask(
            dtm,
            support_mask,
            valley_mask,
            resolution
        )

        local_win_m = float(cfg.get("local_max_window_m", 90.0))
        local_win_cells = max(3, int(round(local_win_m / resolution)))
        if local_win_cells % 2 == 0:
            local_win_cells += 1

        dist_for_max = np.where(
            valid & np.isfinite(valley_dist),
            valley_dist,
            -9999.0
        )

        local_max = ndimage.maximum_filter(
            dist_for_max,
            size=local_win_cells,
            mode="nearest"
        )

        tolerance_m = float(cfg.get("local_max_tolerance_m", 10.0))
        center_band = dist_for_max >= (local_max - tolerance_m)

        tpi = self.compute_tpi_raw(
            dtm,
            support_mask,
            resolution,
            float(cfg.get("tpi_radius_m", 160.0))
        )

        valid_tpi = tpi[valid & np.isfinite(tpi)]
        if valid_tpi.size == 0:
            print("[ridge_center_supplement] no valid TPI")
            return []

        tpi_th = np.percentile(
            valid_tpi,
            float(cfg.get("min_tpi_percentile", 22.0))
        )

        mask = (
            valid
            & center_band
            & np.isfinite(valley_dist)
            & (valley_dist >= float(cfg.get("min_distance_to_valley_m", 16.0)))
            & np.isfinite(hand)
            & (hand >= float(cfg.get("min_hand_m", 0.8)))
            & np.isfinite(tpi)
            & (tpi >= tpi_th)
        )

        if edge_dist is not None:
            mask = mask & (
                edge_dist >= float(cfg.get("min_edge_distance_m", 45.0))
            )

        if existing_ridge_lines:
            ridge_dist = self.compute_line_distance_grid(
                existing_ridge_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )
            mask = mask & (
                ridge_dist >= float(cfg.get("min_distance_from_existing_ridge_m", 30.0))
            )

        closing_disk = int(cfg.get("closing_disk", 2))
        if closing_disk > 0:
            mask = morphology.binary_closing(
                mask,
                morphology.disk(closing_disk)
            )

        mask = mask & valid

        min_area = int(cfg.get("min_area_cells", 25))
        if min_area > 0:
            mask = morphology.remove_small_objects(
                mask,
                min_size=min_area
            )

        lines = self.skeleton_to_lines(
            mask.astype(np.uint8),
            transform,
            "ridge_center_supplement",
            min_length=float(cfg.get("min_line_length", 75.0))
        )

        if not lines:
            print("[ridge_center_supplement] lines=0")
            return []

        scored = []
        rows, cols = dtm.shape

        for line in lines:
            vals_dist = []
            vals_hand = []
            vals_tpi = []

            n_samples = 40
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(valley_dist[r, c]):
                        vals_dist.append(valley_dist[r, c])
                    if np.isfinite(hand[r, c]):
                        vals_hand.append(hand[r, c])
                    if np.isfinite(tpi[r, c]):
                        vals_tpi.append(tpi[r, c])

            mean_dist = float(np.mean(vals_dist)) if vals_dist else 0.0
            mean_hand = float(np.mean(vals_hand)) if vals_hand else 0.0
            mean_tpi = float(np.mean(vals_tpi)) if vals_tpi else 0.0

            score = line.length * (
                0.40 * mean_hand
                + 0.35 * mean_dist
                + 0.25 * max(mean_tpi, 0.0)
            )
            scored.append((line, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        keep_top_n = int(cfg.get("keep_top_n", 70))
        result = [x[0] for x in scored[:keep_top_n]]

        print(f"[ridge_center_supplement] lines={len(result)}")
        return result

    def evaluate_line_extremeness(self, line: LineString, dtm: np.ndarray,
                                  transform: Affine, resolution: float,
                                  mode: str = 'valley',
                                  profile_half_width_m: float = 30.0,
                                  n_line_samples: int = 30,
                                  n_profile_samples: int = 21) -> Tuple[float, float, int]:
        coords = np.array(line.coords)
        if len(coords) < 2:
            return 0.0, 0.0, 0

        rows, cols = dtm.shape
        inv_a = 1.0 / transform.a if transform.a != 0 else 1.0

        def world_to_rc(x, y):
            c = (x - transform.c) / transform.a
            r = (y - transform.f) / transform.e
            return r, c

        def sample_dtm_rc(r, c):
            r0 = int(np.floor(r))
            c0 = int(np.floor(c))
            r1 = r0 + 1
            c1 = c0 + 1
            if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
                return np.nan
            fr = r - r0
            fc = c - c0
            v00 = dtm[r0, c0]
            v01 = dtm[r0, c1]
            v10 = dtm[r1, c0]
            v11 = dtm[r1, c1]
            if not (np.isfinite(v00) and np.isfinite(v01) and np.isfinite(v10) and np.isfinite(v11)):
                return np.nan
            return v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc + v10 * fr * (1 - fc) + v11 * fr * fc

        half_n = n_profile_samples // 2
        profile_dists = np.linspace(-profile_half_width_m, profile_half_width_m, n_profile_samples)

        extreme_count = 0
        relief_sum = 0.0
        valid_count = 0

        for i in range(n_line_samples):
            frac = i / (n_line_samples - 1) if n_line_samples > 1 else 0.0
            pt = line.interpolate(frac, normalized=True)
            px, py = pt.x, pt.y

            if i == 0:
                p_next = line.interpolate(min(frac + 0.05, 1.0), normalized=True)
                dx = p_next.x - px
                dy = p_next.y - py
            elif i == n_line_samples - 1:
                p_prev = line.interpolate(max(frac - 0.05, 0.0), normalized=True)
                dx = px - p_prev.x
                dy = py - p_prev.y
            else:
                p_next = line.interpolate(min(frac + 0.05, 1.0), normalized=True)
                p_prev = line.interpolate(max(frac - 0.05, 0.0), normalized=True)
                dx = p_next.x - p_prev.x
                dy = p_next.y - p_prev.y

            tang_len = np.sqrt(dx * dx + dy * dy)
            if tang_len < 1e-6:
                continue
            tx, ty = dx / tang_len, dy / tang_len
            nx, ny = -ty, tx

            profile_z = []
            for d in profile_dists:
                sx = px + nx * d
                sy = py + ny * d
                sr, sc = world_to_rc(sx, sy)
                z = sample_dtm_rc(sr, sc)
                profile_z.append(z)

            profile_z = np.array(profile_z)
            valid_mask = np.isfinite(profile_z)
            if np.sum(valid_mask) < n_profile_samples // 2:
                continue

            center_z = profile_z[half_n]
            if not np.isfinite(center_z):
                continue

            valid_z = profile_z[valid_mask]
            if mode == 'valley':
                min_z = np.min(valid_z)
                is_extreme = center_z <= min_z + 0.5
                left_valid = profile_z[:half_n]
                right_valid = profile_z[half_n + 1:]
                left_valid = left_valid[np.isfinite(left_valid)]
                right_valid = right_valid[np.isfinite(right_valid)]
                if len(left_valid) > 0 and len(right_valid) > 0:
                    side_mean = (np.mean(left_valid) + np.mean(right_valid)) / 2.0
                    local_relief = side_mean - center_z
                else:
                    local_relief = 0.0
            else:
                max_z = np.max(valid_z)
                is_extreme = center_z >= max_z - 0.5
                left_valid = profile_z[:half_n]
                right_valid = profile_z[half_n + 1:]
                left_valid = left_valid[np.isfinite(left_valid)]
                right_valid = right_valid[np.isfinite(right_valid)]
                if len(left_valid) > 0 and len(right_valid) > 0:
                    side_mean = (np.mean(left_valid) + np.mean(right_valid)) / 2.0
                    local_relief = center_z - side_mean
                else:
                    local_relief = 0.0

            if is_extreme:
                extreme_count += 1
            relief_sum += max(0.0, local_relief)
            valid_count += 1

        if valid_count == 0:
            return 0.0, 0.0, 0

        extreme_ratio = extreme_count / valid_count
        mean_relief = relief_sum / valid_count
        return extreme_ratio, mean_relief, valid_count

    def filter_final_ridge_lines_by_profile(
        self,
        lines: List[LineString],
        dtm: np.ndarray,
        support_mask: np.ndarray,
        transform: Affine,
        ridge_score: Optional[np.ndarray] = None,
        valley_lines: Optional[List[LineString]] = None,
        edge_dist: Optional[np.ndarray] = None,
        cfg: Optional[dict] = None
    ) -> List[LineString]:
        """Final line-level quality filter for the combined ridge result."""
        cfg = cfg or self.config.get("ridge_final_filter", {})
        if not cfg.get("enabled", False) or not lines:
            return lines

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = dtm.shape

        min_length = float(cfg.get("min_line_length_m", 100.0))
        min_mean_score = float(cfg.get("min_mean_score", 0.38))
        profile_hw = float(cfg.get("profile_half_width_m", 45.0))
        min_extreme_ratio = float(cfg.get("min_profile_extreme_ratio", 0.25))
        min_profile_relief = float(cfg.get("min_profile_relief_m", 0.5))
        min_valley_dist = float(cfg.get("min_distance_to_valley_m", 8.0))
        max_near_valley_ratio = float(cfg.get("max_near_valley_ratio", 0.35))
        min_edge_dist = float(cfg.get("min_edge_distance_m", 45.0))
        max_near_edge_ratio = float(
            cfg.get(
                "max_near_edge_ratio",
                self.config.get("edge_filter", {}).get("max_near_edge_ratio", 0.25)
            )
        )

        valley_dist = None
        if valley_lines and min_valley_dist > 0:
            valley_dist = self.compute_line_distance_grid(
                valley_lines,
                dtm.shape,
                transform,
                support_mask,
                resolution
            )

        kept = []
        n_samples = int(cfg.get("n_line_samples", 40))

        for line in lines:
            if line is None or line.is_empty or line.length < min_length:
                continue

            score_vals = []
            near_valley_count = 0
            edge_near_count = 0
            valid_count = 0

            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if not (0 <= r < rows and 0 <= c < cols):
                    continue

                valid_count += 1

                if ridge_score is not None:
                    s = ridge_score[r, c]
                    if np.isfinite(s):
                        score_vals.append(s)

                if valley_dist is not None:
                    vd = valley_dist[r, c]
                    if np.isfinite(vd) and vd < min_valley_dist:
                        near_valley_count += 1

                if edge_dist is not None and edge_dist[r, c] < min_edge_dist:
                    edge_near_count += 1

            if valid_count == 0:
                continue

            if ridge_score is not None:
                if not score_vals:
                    continue
                mean_score = float(np.mean(score_vals))
                if mean_score < min_mean_score:
                    continue

            if valley_dist is not None:
                near_valley_ratio = near_valley_count / valid_count
                if near_valley_ratio > max_near_valley_ratio:
                    continue

            if edge_dist is not None:
                near_edge_ratio = edge_near_count / valid_count
                if near_edge_ratio > max_near_edge_ratio:
                    continue

            er, relief, vc = self.evaluate_line_extremeness(
                line,
                dtm,
                transform,
                resolution,
                mode="ridge",
                profile_half_width_m=profile_hw,
                n_line_samples=30,
                n_profile_samples=21
            )
            if vc == 0 or er < min_extreme_ratio or relief < min_profile_relief:
                continue

            kept.append(line)

        print(f"[ridge_final_filter] {len(lines)} -> {len(kept)}")
        return kept

    def prune_dense_ridge_lines(
        self,
        lines: List[LineString],
        dtm: np.ndarray,
        transform: Affine,
        ridge_score: Optional[np.ndarray],
        cfg: dict
    ) -> List[LineString]:
        """Remove close, nearly parallel weaker ridge lines."""
        if not cfg.get("enabled", False) or not lines or len(lines) <= 1:
            return lines

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = dtm.shape
        min_distance = float(cfg.get("min_distance_m", 20.0))
        near_ratio_threshold = float(cfg.get("near_ratio_threshold", 0.60))
        max_angle_diff = float(cfg.get("max_angle_diff_deg", 25.0))
        n_samples = int(cfg.get("n_samples", 30))

        def line_direction(line: LineString):
            coords = np.array(line.coords)
            if len(coords) < 2:
                return None
            vec = coords[-1] - coords[0]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return None
            return vec / norm

        def undirected_angle(v1, v2):
            if v1 is None or v2 is None:
                return 0.0
            angle = self.angle_between_vectors(v1, v2)
            return min(angle, 180.0 - angle)

        def mean_score(line: LineString):
            if ridge_score is None:
                return 0.0
            vals = []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)
                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)
                if 0 <= r < rows and 0 <= c < cols and np.isfinite(ridge_score[r, c]):
                    vals.append(ridge_score[r, c])
            return float(np.mean(vals)) if vals else 0.0

        scored = []
        for line in lines:
            if line is None or line.is_empty:
                continue
            er, relief, valid_count = self.evaluate_line_extremeness(
                line,
                dtm,
                transform,
                resolution,
                mode="ridge",
                profile_half_width_m=float(cfg.get("profile_half_width_m", 60.0)),
                n_line_samples=20,
                n_profile_samples=21
            )
            quality = (
                0.45 * mean_score(line)
                + 0.25 * min(line.length / 500.0, 1.0)
                + 0.20 * max(relief, 0.0) / (max(relief, 0.0) + 3.0)
                + 0.10 * er
            )
            scored.append({
                "line": line,
                "quality": float(quality),
                "direction": line_direction(line)
            })

        scored.sort(key=lambda item: item["quality"], reverse=True)
        selected = []

        for item in scored:
            line = item["line"]
            keep = True
            for kept in selected:
                if undirected_angle(item["direction"], kept["direction"]) > max_angle_diff:
                    continue

                near_count = 0
                valid_count = 0
                for i in range(n_samples):
                    frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                    pt = line.interpolate(frac, normalized=True)
                    valid_count += 1
                    if pt.distance(kept["line"]) < min_distance:
                        near_count += 1

                if valid_count > 0 and near_count / valid_count >= near_ratio_threshold:
                    keep = False
                    break

            if keep:
                selected.append(item)

        result = [item["line"] for item in selected]
        print(f"[ridge_dense_prune] {len(lines)} -> {len(result)}")
        return result

    def evaluate_lines_importance(self, lines: List[LineString], dtm: np.ndarray,
                                  transform: Affine, accumulation: np.ndarray,
                                  support_mask: np.ndarray, mode: str) -> List[dict]:
        imp_cfg = self.config.get('line_importance', {})
        profile_hw = float(imp_cfg.get('profile_half_width_m', 30.0))
        n_line_s = int(imp_cfg.get('n_line_samples', 30))
        n_prof_s = int(imp_cfg.get('n_profile_samples', 21))

        resolution = abs(transform.a) if transform.a != 0 else 1.0
        rows, cols = dtm.shape

        raw_scores = []
        raw_lengths = []
        raw_extreme = []
        raw_relief = []
        raw_acc_vals = []

        for line in lines:
            if line.is_empty or line.length < 1.0:
                raw_scores.append({'extreme_ratio': 0.0, 'mean_relief': 0.0,
                                   'importance_score': 0.0, 'importance_level': 'low'})
                raw_lengths.append(0.0)
                raw_extreme.append(0.0)
                raw_relief.append(0.0)
                raw_acc_vals.append(0.0)
                continue

            er, mr, vc = self.evaluate_line_extremeness(
                line, dtm, transform, resolution, mode,
                profile_half_width_m=profile_hw,
                n_line_samples=n_line_s,
                n_profile_samples=n_prof_s
            )

            coords = np.array(line.coords)
            acc_vals = []
            for pt_x, pt_y in coords:
                c = int((pt_x - transform.c) / transform.a)
                r = int((pt_y - transform.f) / transform.e)
                if 0 <= r < rows and 0 <= c < cols:
                    if support_mask[r, c] > 0 and np.isfinite(accumulation[r, c]):
                        acc_vals.append(accumulation[r, c])
            mean_acc = np.mean(acc_vals) if acc_vals else 0.0

            raw_lengths.append(line.length)
            raw_extreme.append(er)
            raw_relief.append(mr)
            raw_acc_vals.append(mean_acc)

        if not lines:
            return []

        def safe_norm(arr):
            arr = np.array(arr, dtype=float)
            if len(arr) == 0:
                return arr
            p10 = np.percentile(arr, 10)
            p90 = np.percentile(arr, 90)
            spread = p90 - p10
            if spread < 1e-6:
                return np.zeros_like(arr)
            return np.clip((arr - p10) / spread, 0, 1)

        norm_acc = safe_norm(raw_acc_vals)
        norm_len = safe_norm(raw_lengths)

        if mode == 'valley':
            bv_cfg = self.config.get('broad_valley', {})
            importance_radius = float(bv_cfg.get('radius_m', 120.0))
            tpi_arr_for_importance = self.compute_tpi_score(
                dtm,
                support_mask,
                resolution,
                importance_radius
            )
        else:
            br_cfg = self.config.get('broad_ridge', {})
            importance_radius = float(br_cfg.get('radius_m', 50.0))
            tpi_arr_for_importance = self.compute_tpi_raw(
                dtm,
                support_mask,
                resolution,
                importance_radius
            )

        raw_tpi_vals = []
        for i, line in enumerate(lines):
            if line.is_empty:
                raw_tpi_vals.append(0.0)
                continue
            mid = line.interpolate(0.5, normalized=True)
            c = int((mid.x - transform.c) / transform.a)
            r = int((mid.y - transform.f) / transform.e)
            if 0 <= r < rows and 0 <= c < cols:
                val = tpi_arr_for_importance[r, c]
                raw_tpi_vals.append(float(val) if np.isfinite(val) else 0.0)
            else:
                raw_tpi_vals.append(0.0)

        if mode == 'valley':
            norm_tpi = safe_norm([-v for v in raw_tpi_vals])
            norm_acc_for_score = safe_norm(raw_acc_vals)
        else:
            norm_tpi = safe_norm(raw_tpi_vals)
            norm_acc_for_score = safe_norm([-v for v in raw_acc_vals])

        results = []
        high_th = float(imp_cfg.get('high_threshold', 0.70))
        med_th = float(imp_cfg.get('medium_threshold', 0.45))
        val_er_min = float(imp_cfg.get('valley_extreme_ratio_min', 0.60))
        val_rel_min = float(imp_cfg.get('valley_relief_min', 2.0))
        rid_er_min = float(imp_cfg.get('ridge_extreme_ratio_min', 0.55))
        rid_rel_min = float(imp_cfg.get('ridge_relief_min', 2.0))

        for i, line in enumerate(lines):
            if mode == 'valley':
                score = (0.35 * norm_acc_for_score[i]
                         + 0.25 * norm_tpi[i]
                         + 0.25 * raw_extreme[i]
                         + 0.15 * norm_len[i])
                er_min = val_er_min
                rel_min = val_rel_min
            else:
                score = (0.30 * norm_acc_for_score[i]
                         + 0.25 * norm_tpi[i]
                         + 0.25 * raw_extreme[i]
                         + 0.20 * norm_len[i])
                er_min = rid_er_min
                rel_min = rid_rel_min

            if raw_extreme[i] < er_min or raw_relief[i] < rel_min:
                score *= 0.5

            if score >= high_th:
                level = 'high'
            elif score >= med_th:
                level = 'medium'
            else:
                level = 'low'

            results.append({
                'extreme_ratio': round(raw_extreme[i], 3),
                'mean_local_relief': round(raw_relief[i], 2),
                'importance_score': round(float(score), 3),
                'importance_level': level
            })

        return results

    def line_endpoint_direction(self, coords: np.ndarray, at_start: bool = True, n: int = 3) -> Optional[np.ndarray]:
        """
        计算端点方向向量
        """
        if coords is None or len(coords) < 2:
            return None

        if at_start:
            idx = min(n, len(coords) - 1)
            v = np.array(coords[idx]) - np.array(coords[0])
        else:
            idx = max(0, len(coords) - 1 - n)
            v = np.array(coords[-1]) - np.array(coords[idx])

        norm = np.linalg.norm(v)
        if norm == 0:
            return None
        return v / norm

    def angle_between_vectors(self, v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
        """
        计算两向量夹角（度）
        """
        if v1 is None or v2 is None:
            return 0.0
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def can_merge_lines(self, line1: LineString, line2: LineString,
                        merge_distance: float, max_angle_deg: float) -> Optional[List[List[float]]]:
        """
        判断两线是否可合并，返回合并后的坐标或 None
        """
        coords1 = np.array(line1.coords)
        coords2 = np.array(line2.coords)
        if len(coords1) < 2 or len(coords2) < 2:
            return None

        s1, e1 = coords1[0], coords1[-1]
        s2, e2 = coords2[0], coords2[-1]

        candidates = [
            ("end-start", e1, s2),
            ("start-end", s1, e2),
            ("start-start", s1, s2),
            ("end-end", e1, e2)
        ]

        best = None
        best_dist = None
        for mode, p1, p2 in candidates:
            dist = float(np.linalg.norm(p1 - p2))
            if dist > merge_distance:
                continue

            if max_angle_deg < 180:
                if mode == "end-start":
                    v1 = self.line_endpoint_direction(coords1, at_start=False)
                    v2 = self.line_endpoint_direction(coords2, at_start=True)
                elif mode == "start-end":
                    v1 = self.line_endpoint_direction(coords2, at_start=False)
                    v2 = self.line_endpoint_direction(coords1, at_start=True)
                elif mode == "start-start":
                    v1 = self.line_endpoint_direction(coords1, at_start=True)
                    v2 = self.line_endpoint_direction(coords2, at_start=True)
                    if v1 is not None:
                        v1 = -v1
                else:  # end-end
                    v1 = self.line_endpoint_direction(coords1, at_start=False)
                    v2 = self.line_endpoint_direction(coords2, at_start=False)
                    if v2 is not None:
                        v2 = -v2

                angle = self.angle_between_vectors(v1, v2)
                if angle > max_angle_deg:
                    continue

            if best is None or dist < best_dist:
                best = mode
                best_dist = dist

        if best is None:
            return None

        if best == "end-start":
            merged = np.vstack([coords1, coords2])
        elif best == "start-end":
            merged = np.vstack([coords2, coords1])
        elif best == "start-start":
            merged = np.vstack([coords1[::-1], coords2])
        else:
            merged = np.vstack([coords1, coords2[::-1]])

        # 去重连接点
        if len(merged) > 1:
            cleaned = [merged[0]]
            for pt in merged[1:]:
                if np.linalg.norm(pt - cleaned[-1]) > 1e-6:
                    cleaned.append(pt)
            return [pt.tolist() for pt in cleaned]

        return None

    def postprocess_lines(self, lines: List[LineString], merge_distance: float,
                          simplify_tolerance: float, max_merge_angle_deg: float,
                          max_merge_iterations: int) -> List[LineString]:
        """
        简单后处理：端点合并 + 线简化
        """
        if not lines:
            return []

        merged_lines = list(lines)
        if merge_distance > 0 and len(merged_lines) > 1:
            for _ in range(max(1, max_merge_iterations)):
                merged = False
                used = [False] * len(merged_lines)
                new_lines = []

                endpoint_points = []
                endpoint_line_ids = []
                for idx, line in enumerate(merged_lines):
                    if line is None or line.is_empty:
                        continue
                    coords = np.array(line.coords)
                    if len(coords) < 2:
                        continue
                    endpoint_points.append(coords[0])
                    endpoint_line_ids.append(idx)
                    endpoint_points.append(coords[-1])
                    endpoint_line_ids.append(idx)

                endpoint_tree = None
                if endpoint_points:
                    endpoint_points = np.asarray(endpoint_points, dtype=float)
                    endpoint_line_ids = np.asarray(endpoint_line_ids, dtype=np.int32)
                    endpoint_tree = cKDTree(endpoint_points)

                for i, line in enumerate(merged_lines):
                    if used[i]:
                        continue
                    current = line

                    next_j_min = i + 1
                    while endpoint_tree is not None:
                        if current is None or current.is_empty:
                            break
                        current_coords = np.array(current.coords)
                        if len(current_coords) < 2:
                            break

                        endpoint_hits = endpoint_tree.query_ball_point(
                            [current_coords[0], current_coords[-1]],
                            merge_distance
                        )
                        if len(endpoint_hits) > 0:
                            hit_ids = set()
                            for hits in endpoint_hits:
                                hit_ids.update(hits)
                            candidate_ids = sorted({
                                int(endpoint_line_ids[h])
                                for h in hit_ids
                                if next_j_min <= int(endpoint_line_ids[h]) < len(merged_lines)
                                and not used[int(endpoint_line_ids[h])]
                            })
                        else:
                            candidate_ids = []

                        did_merge_current = False
                        for j in candidate_ids:
                            merged_coords = self.can_merge_lines(
                                current,
                                merged_lines[j],
                                merge_distance,
                                max_merge_angle_deg
                            )
                            if merged_coords is not None:
                                current = LineString(merged_coords)
                                used[j] = True
                                merged = True
                                next_j_min = j + 1
                                did_merge_current = True
                                break

                        if not did_merge_current:
                            break

                    used[i] = True
                    if not current.is_empty:
                        new_lines.append(current)

                merged_lines = new_lines
                if not merged:
                    break

        if simplify_tolerance <= 0:
            return merged_lines

        processed = []
        for line in merged_lines:
            simplified = line.simplify(simplify_tolerance, preserve_topology=False)
            if not simplified.is_empty:
                processed.append(simplified)
        return processed

    def skeleton_to_lines(self, mask: np.ndarray, transform: Affine, 
                         feature_type: str, min_length: float = None) -> List[LineString]:
        """
        将二值掩膜骨架化为矢量线，基于图结构追踪
        """
        if min_length is None:
            min_length = self.config['extraction']['min_line_length']
        
        # 骨架化
        skeleton = morphology.skeletonize(mask > 0)
        
        rows, cols = skeleton.shape
        
        # 找出所有骨架像元
        skeleton_coords = np.column_stack(np.where(skeleton))
        if len(skeleton_coords) == 0:
            print(f"[✓] 骨架化为 0 条 {feature_type} 线")
            return []

        coords_list = [tuple(coord) for coord in skeleton_coords]
        coord_to_idx = {coord: idx for idx, coord in enumerate(coords_list)}

        # 为每个骨架像元建立邻接关系
        neighbors = [[] for _ in range(len(coords_list))]
        for idx, (r, c) in enumerate(coords_list):
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    neighbor_idx = coord_to_idx.get((nr, nc))
                    if neighbor_idx is not None:
                        neighbors[idx].append(neighbor_idx)

        degree = [len(neighbors[idx]) for idx in range(len(coords_list))]
        node_indices = set([idx for idx, deg in enumerate(degree) if deg != 2])

        def edge_key(a: int, b: int) -> Tuple[int, int]:
            return (a, b) if a < b else (b, a)

        def trace_path(start_idx: int, next_idx: int) -> List[int]:
            path = [start_idx, next_idx]
            prev_idx = start_idx
            curr_idx = next_idx

            while True:
                if curr_idx in node_indices and curr_idx != start_idx:
                    break

                candidates = [n for n in neighbors[curr_idx] if n != prev_idx]
                if not candidates:
                    break

                next_candidate = None
                for cand in candidates:
                    if edge_key(curr_idx, cand) not in visited_edges:
                        next_candidate = cand
                        break

                if next_candidate is None:
                    break

                visited_edges.add(edge_key(curr_idx, next_candidate))
                path.append(next_candidate)
                prev_idx, curr_idx = curr_idx, next_candidate

                if curr_idx == start_idx:
                    break

            return path

        visited_edges = set()
        lines = []

        # 从端点/分叉点出发追踪
        for start_idx in node_indices:
            for neighbor_idx in neighbors[start_idx]:
                edge = edge_key(start_idx, neighbor_idx)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                path_indices = trace_path(start_idx, neighbor_idx)

                if len(path_indices) < 2:
                    continue

                geo_coords = []
                for idx in path_indices:
                    r, c = coords_list[idx]
                    x = transform.c + (c + 0.5) * transform.a
                    y = transform.f + (r + 0.5) * transform.e
                    geo_coords.append([x, y])

                geo_coords = np.array(geo_coords)
                distances = np.sqrt(np.sum(np.diff(geo_coords, axis=0) ** 2, axis=1))
                total_length = np.sum(distances)

                if total_length >= min_length:
                    lines.append(LineString(geo_coords))

        # 处理闭环骨架（所有度数为 2）
        for start_idx in range(len(coords_list)):
            for neighbor_idx in neighbors[start_idx]:
                edge = edge_key(start_idx, neighbor_idx)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                path = [start_idx, neighbor_idx]
                prev_idx = start_idx
                curr_idx = neighbor_idx

                while True:
                    candidates = [n for n in neighbors[curr_idx] if n != prev_idx]
                    if not candidates:
                        break

                    next_candidate = None
                    for cand in candidates:
                        if edge_key(curr_idx, cand) not in visited_edges:
                            next_candidate = cand
                            break

                    if next_candidate is None:
                        break

                    visited_edges.add(edge_key(curr_idx, next_candidate))
                    path.append(next_candidate)
                    prev_idx, curr_idx = curr_idx, next_candidate

                    if curr_idx == start_idx:
                        break

                if len(path) < 2:
                    continue

                geo_coords = []
                for idx in path:
                    r, c = coords_list[idx]
                    x = transform.c + (c + 0.5) * transform.a
                    y = transform.f + (r + 0.5) * transform.e
                    geo_coords.append([x, y])

                geo_coords = np.array(geo_coords)
                distances = np.sqrt(np.sum(np.diff(geo_coords, axis=0) ** 2, axis=1))
                total_length = np.sum(distances)

                if total_length >= min_length:
                    lines.append(LineString(geo_coords))

        print(f"[✓] 骨架化为 {len(lines)} 条 {feature_type} 线（使用完整路径）")
        return lines

    def save_terrain_features_geojson(self, valley_lines: List[LineString], 
                                      ridge_lines: List[LineString],
                                      valley_method_map: Optional[Dict[int, str]] = None,
                                      ridge_method_map: Optional[Dict[int, str]] = None,
                                      valley_importance: Optional[List[dict]] = None,
                                      ridge_importance: Optional[List[dict]] = None):
        """
        保存山谷线和山脊线为 GeoJSON
        """
        features = []
        
        for idx, line in enumerate(valley_lines):
            if not line.is_empty:
                method = "flow_trace"
                if valley_method_map and idx in valley_method_map:
                    method = valley_method_map[idx]
                props = {
                    "feature_type": "valley",
                    "valley_method": method
                }
                if valley_importance and idx < len(valley_importance):
                    imp = valley_importance[idx]
                    props["importance_score"] = imp.get("importance_score", 0)
                    props["extreme_ratio"] = imp.get("extreme_ratio", 0)
                    props["mean_local_relief"] = imp.get("mean_local_relief", 0)
                    props["importance_level"] = imp.get("importance_level", "low")
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": list(line.coords)
                    },
                    "properties": props
                }
                features.append(feature)
        
        for idx, line in enumerate(ridge_lines):
            if not line.is_empty:
                method = "flow_trace"
                if ridge_method_map and idx in ridge_method_map:
                    method = ridge_method_map[idx]
                props = {
                    "feature_type": "ridge",
                    "ridge_method": method
                }
                if ridge_importance and idx < len(ridge_importance):
                    imp = ridge_importance[idx]
                    props["importance_score"] = imp.get("importance_score", 0)
                    props["extreme_ratio"] = imp.get("extreme_ratio", 0)
                    props["mean_local_relief"] = imp.get("mean_local_relief", 0)
                    props["importance_level"] = imp.get("importance_level", "low")
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": list(line.coords)
                    },
                    "properties": props
                }
                features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        output_path = os.path.join(self.output_dir, 'terrain_features.geojson')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        print(f"[✓] 保存 GeoJSON：{output_path}")

    def rasterize_lines(self, lines: List[LineString], shape: Tuple[int, int],
                        transform: Affine) -> np.ndarray:
        """
        将矢量线栅格化为二值掩膜
        """
        if not lines:
            return np.zeros(shape, dtype=np.uint8)

        shapes = [(line, 1) for line in lines if not line.is_empty]
        if not shapes:
            return np.zeros(shape, dtype=np.uint8)

        return rio_features.rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

    def compute_line_distance_grid(self, lines: List[LineString], shape: Tuple[int, int],
                                   transform: Affine, support_mask: np.ndarray,
                                   resolution: float) -> np.ndarray:
        """
        计算每个栅格像元到最近线的距离（米）
        """
        line_mask = self.rasterize_lines(lines, shape, transform)
        if np.sum(line_mask) == 0:
            dist = np.full(shape, np.inf, dtype=np.float32)
        else:
            dist = ndimage.distance_transform_edt(line_mask == 0, sampling=resolution).astype(np.float32)

        dist[support_mask == 0] = np.inf
        return dist

    def compute_distance_and_relief_from_mask(
        self,
        dtm: np.ndarray,
        support_mask: np.ndarray,
        valley_mask: np.ndarray,
        resolution: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid = (support_mask > 0) & np.isfinite(dtm)
        valley_mask = valley_mask.astype(bool) & valid

        if np.sum(valley_mask) == 0:
            dist = np.full(dtm.shape, np.inf, dtype=np.float32)
            relief = np.zeros_like(dtm, dtype=np.float32)
            dist[~valid] = np.nan
            relief[~valid] = np.nan
            return dist, relief

        dist, inds = ndimage.distance_transform_edt(
            ~valley_mask,
            sampling=resolution,
            return_indices=True
        )

        nearest_valley_z = dtm[inds[0], inds[1]]
        relief = dtm - nearest_valley_z

        dist = dist.astype(np.float32)
        relief = relief.astype(np.float32)
        dist[~valid] = np.nan
        relief[~valid] = np.nan
        return dist, relief

    def compute_relief_to_valley(self, dtm: np.ndarray, support_mask: np.ndarray,
                                 transform: Affine, valley_lines: List[LineString]) -> np.ndarray:
        """
        计算每个像元相对最近山谷线的高差。
        relief_to_valley 越大，越可能是真正山脊或高地分水岭。
        """
        resolution = abs(transform.a) if transform.a != 0 else 1.0

        valley_mask = self.rasterize_lines(
            valley_lines,
            dtm.shape,
            transform
        )

        if np.sum(valley_mask) == 0:
            relief = np.zeros_like(dtm, dtype=np.float32)
            relief[(support_mask == 0) | ~np.isfinite(dtm)] = np.nan
            return relief

        dist, inds = ndimage.distance_transform_edt(
            valley_mask == 0,
            sampling=resolution,
            return_indices=True
        )

        nearest_r = inds[0]
        nearest_c = inds[1]

        nearest_valley_z = dtm[nearest_r, nearest_c]
        relief = dtm - nearest_valley_z

        relief[(support_mask == 0) | ~np.isfinite(dtm)] = np.nan
        return relief.astype(np.float32)

    def compute_edge_distance(self, support_mask: np.ndarray, resolution: float) -> np.ndarray:
        valid = support_mask > 0
        edge_dist = ndimage.distance_transform_edt(valid, sampling=resolution).astype(np.float32)
        edge_dist[~valid] = 0.0
        return edge_dist

    def filter_lines_by_edge_distance(
        self,
        lines: List[LineString],
        edge_dist: np.ndarray,
        transform: Affine,
        min_edge_distance_m: float = 50.0,
        max_near_edge_ratio: float = 0.25,
        n_samples: int = 30,
        reject_if_endpoint_near_edge: bool = False,
        endpoint_edge_buffer_m: Optional[float] = None,
        short_line_length_m: float = 120.0,
        short_line_near_edge_ratio: float = 0.05
    ) -> List[LineString]:
        if edge_dist is None or not lines:
            return lines

        rows, cols = edge_dist.shape
        kept = []
        endpoint_edge_buffer_m = (
            min_edge_distance_m if endpoint_edge_buffer_m is None else endpoint_edge_buffer_m
        )

        for line in lines:
            if line.is_empty or line.length <= 0:
                continue

            if reject_if_endpoint_near_edge:
                coords = np.array(line.coords)
                endpoint_near = False
                for x, y in (coords[0], coords[-1]):
                    c = int((x - transform.c) / transform.a)
                    r = int((y - transform.f) / transform.e)
                    if not (0 <= r < rows and 0 <= c < cols):
                        endpoint_near = True
                        break
                    if edge_dist[r, c] < endpoint_edge_buffer_m:
                        endpoint_near = True
                        break
                if endpoint_near:
                    continue

            near_count = 0
            valid_count = 0

            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    valid_count += 1
                    if edge_dist[r, c] < min_edge_distance_m:
                        near_count += 1

            if valid_count == 0:
                continue

            near_ratio = near_count / valid_count

            if line.length < short_line_length_m and near_ratio > short_line_near_edge_ratio:
                continue

            if near_ratio <= max_near_edge_ratio:
                kept.append(line)

        return kept

    def filter_lines_by_length(self, lines: List[LineString], min_length: float) -> List[LineString]:
        if not lines:
            return []
        return [ln for ln in lines if (not ln.is_empty and ln.length >= min_length)]

    def merge_ridge_network_lines(
        self,
        core_lines: List[LineString],
        connector_lines: List[LineString],
        snap_tolerance: float = 6.0,
        min_length: float = 120.0
    ) -> List[LineString]:
        """
        将山脊候选线和 gap connector 合并成连续山脊网络。
        注意：connector 两端必须已接到原始线端点。
        """
        all_lines = [
            ln for ln in (core_lines + connector_lines)
            if ln is not None and not ln.is_empty and ln.length > 0
        ]

        if not all_lines:
            return []

        geom = unary_union(all_lines)
        geom = snap(geom, geom, snap_tolerance)
        merged = linemerge(geom)

        result = []
        if merged.geom_type == "LineString":
            result = [merged]
        elif merged.geom_type == "MultiLineString":
            result = list(merged.geoms)
        elif merged.geom_type == "GeometryCollection":
            for g in merged.geoms:
                if g.geom_type == "LineString":
                    result.append(g)
                elif g.geom_type == "MultiLineString":
                    result.extend(list(g.geoms))

        result = [
            ln for ln in result
            if ln is not None and not ln.is_empty and ln.length >= min_length
        ]

        return result

    def filter_final_ridge_network(
        self,
        lines: List[LineString],
        dtm: np.ndarray,
        transform: Affine,
        ridge_score: np.ndarray,
        valley_dist: np.ndarray,
        resolution: float,
        min_length: float = 120.0,
        min_mean_score: float = 0.40,
        min_valley_dist: float = 8.0,
        max_near_valley_ratio: float = 0.25,
        min_extreme_ratio: float = 0.05,
        min_relief: float = 0.05
    ) -> List[LineString]:
        """
        对最终合并后的山脊网络做统一质量过滤。
        """
        if not lines:
            return []

        kept = []
        rows, cols = ridge_score.shape

        for line in lines:
            if line.is_empty or line.length < min_length:
                continue

            score_vals = []
            vdist_vals = []

            n_samples = 40

            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = line.interpolate(frac, normalized=True)

                c = int((pt.x - transform.c) / transform.a)
                r = int((pt.y - transform.f) / transform.e)

                if 0 <= r < rows and 0 <= c < cols:
                    if np.isfinite(ridge_score[r, c]):
                        score_vals.append(ridge_score[r, c])
                    if np.isfinite(valley_dist[r, c]):
                        vdist_vals.append(valley_dist[r, c])

            if not score_vals:
                continue

            mean_score = float(np.mean(score_vals))
            if mean_score < min_mean_score:
                continue

            if vdist_vals:
                vdist_vals = np.array(vdist_vals)
                near_ratio = float(np.mean(vdist_vals < min_valley_dist))
                if near_ratio > max_near_valley_ratio:
                    continue

            er, relief, vc = self.evaluate_line_extremeness(
                line,
                dtm,
                transform,
                resolution,
                mode="ridge",
                profile_half_width_m=50.0,
                n_line_samples=30,
                n_profile_samples=21
            )

            if er < min_extreme_ratio or relief < min_relief:
                continue

            kept.append(line)

        return kept

    def map_lines_to_points(self, valley_lines: List[LineString], 
                            ridge_lines: List[LineString],
                            transform: Affine, dtm_shape: Tuple[int, int],
                            support_mask: np.ndarray) -> Dict[int, int]:
        """
        将山谷线和山脊线映射回原始点云
        基于栅格距离（近似）
        返回字典：{点的索引：类型（1=valley，2=ridge）}
        """
        las = self.las_all
        mask_ground = las.classification == self.ground_class
        indices_ground = np.where(mask_ground)[0]
        
        point_type = {}  # 原始点索引 -> 类型（1=valley, 2=ridge）
        buffer_distance = self.config['point_mapping']['point_buffer_distance']
        
        # 栅格距离场
        resolution = self.config['dtm']['resolution']
        valley_dist = self.compute_line_distance_grid(valley_lines, dtm_shape, transform, support_mask, resolution)
        ridge_dist = self.compute_line_distance_grid(ridge_lines, dtm_shape, transform, support_mask, resolution)

        # 对每个地面点，查找对应栅格距离
        ground_x = las.x[mask_ground]
        ground_y = las.y[mask_ground]

        col = ((ground_x - transform.c) / transform.a).astype(np.int64)
        row = ((ground_y - transform.f) / transform.e).astype(np.int64)

        rows, cols = dtm_shape
        valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)

        dist_to_valley = np.full(len(ground_x), np.inf, dtype=np.float32)
        dist_to_ridge = np.full(len(ground_x), np.inf, dtype=np.float32)
        dist_to_valley[valid] = valley_dist[row[valid], col[valid]]
        dist_to_ridge[valid] = ridge_dist[row[valid], col[valid]]

        in_valley = dist_to_valley <= buffer_distance
        in_ridge = dist_to_ridge <= buffer_distance

        both = in_valley & in_ridge
        valley_only = in_valley & ~in_ridge
        ridge_only = in_ridge & ~in_valley

        labels = np.zeros(len(ground_x), dtype=np.uint8)
        labels[valley_only] = 1
        labels[ridge_only] = 2
        labels[both] = np.where(dist_to_valley[both] <= dist_to_ridge[both], 1, 2)

        selected = labels > 0
        selected_idx = np.where(selected)[0]
        for i in selected_idx:
            point_type[indices_ground[i]] = int(labels[i])
        
        valley_count = len([v for v in point_type.values() if v == 1])
        ridge_count = len([v for v in point_type.values() if v == 2])
        print(f"[✓] 映射到原始点云：valley={valley_count}, ridge={ridge_count}")
        return point_type

    def save_feature_points_las(self, point_type: Dict[int, int]):
        """
        保存标记过的特征点为 LAS 文件
        """
        las = self.las_all
        
        # 获取特征点索引
        feature_indices = np.array(list(point_type.keys()), dtype=np.int64)
        
        # 从原始点中提取特征点
        feature_points = las.points[feature_indices].copy()
        
        # 创建新的 LAS 对象
        output_las = laspy.LasData(self.make_output_las_header(las))
        output_las.points = feature_points
        
        # 设置 user_data 字段
        output_las.user_data = np.array([point_type[idx] for idx in feature_indices], dtype=np.uint8)
        
        output_path = os.path.join(self.output_dir, 'terrain_feature_points.las')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_las.write(output_path)
        
        print(f"[✓] 保存特征点 LAS：{output_path}")

    def _render_hillshade(self, dtm: np.ndarray, support_mask: np.ndarray):
        dtm_vis = dtm.copy()
        dtm_vis[support_mask == 0] = np.nan
        with np.errstate(invalid='ignore'):
            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.hillshade(dtm_vis, vert_exag=0.1)
        shaded[support_mask == 0] = 1.0
        return shaded

    def _plot_lines_on_ax(self, ax, lines, transform, color, linewidth, label):
        plotted = False
        for line in lines:
            if not line.is_empty:
                coords = np.array(line.coords)
                col = (coords[:, 0] - transform.c) / transform.a
                row = (coords[:, 1] - transform.f) / transform.e
                lbl = label if not plotted else ''
                ax.plot(col, row, color=color, linewidth=linewidth, label=lbl)
                plotted = True

    def create_preview(self, dtm: np.ndarray, transform: Affine, support_mask: np.ndarray,
                      valley_lines: List[LineString], ridge_lines: List[LineString],
                      flow_lines: Optional[List[LineString]] = None,
                      broad_lines: Optional[List[LineString]] = None):
        shaded = self._render_hillshade(dtm, support_mask)
        resolution = self.config['dtm']['resolution']

        fig, ax = plt.subplots(figsize=(14, 12), dpi=100)
        ax.imshow(shaded, cmap='gray', vmin=0, vmax=1)

        if flow_lines is not None and broad_lines is not None:
            self._plot_lines_on_ax(ax, flow_lines, transform, 'blue', 2.0, 'Valley (flow_trace)')
            self._plot_lines_on_ax(ax, broad_lines, transform, 'cyan', 3.0, 'Valley (broad)')
        else:
            self._plot_lines_on_ax(ax, valley_lines, transform, 'blue', 2.5, 'Valley')

        self._plot_lines_on_ax(ax, ridge_lines, transform, 'red', 2.5, 'Ridge')

        method = self.config.get('extraction', {}).get('method', 'skeleton')
        bv_cfg = self.config.get('broad_valley', {})
        bv_info = f", broad_r={bv_cfg.get('radius_m', '')}m" if bv_cfg.get('enabled') else ''
        title = f'Terrain Features (DTM={resolution}m, method={method}{bv_info})'
        ax.set_xlabel('Column (pixel)', fontsize=10)
        ax.set_ylabel('Row (pixel)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(False)

        output_path = os.path.join(self.output_dir, 'preview.png')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[✓] 保存预览图：{output_path}")

    def create_preview_single(self, dtm: np.ndarray, transform: Affine, support_mask: np.ndarray,
                              lines: List[LineString], ridge_lines: List[LineString],
                              line_color: str, line_label: str, filename: str, title: str):
        shaded = self._render_hillshade(dtm, support_mask)

        fig, ax = plt.subplots(figsize=(14, 12), dpi=100)
        ax.imshow(shaded, cmap='gray', vmin=0, vmax=1)

        self._plot_lines_on_ax(ax, lines, transform, line_color, 2.5, line_label)
        self._plot_lines_on_ax(ax, ridge_lines, transform, 'red', 2.0, 'Ridge')

        ax.set_xlabel('Column (pixel)', fontsize=10)
        ax.set_ylabel('Row (pixel)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(False)

        output_path = os.path.join(self.output_dir, filename)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[✓] 保存预览图：{output_path}")

    def save_debug_images(self, dtm: np.ndarray, support_mask: np.ndarray,
                         valley_accum: np.ndarray, valley_mask: np.ndarray, valley_skeleton: np.ndarray,
                         ridge_accum: np.ndarray, ridge_mask: np.ndarray, ridge_skeleton: np.ndarray,
                         transform: Affine):
        """
        保存调试图像
        """
        if not self.config.get('output', {}).get('save_debug_images', False):
            return
        
        import matplotlib.pyplot as plt
        
        # DTM
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(dtm, cmap='viridis')
        ax.set_title('DTM')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_dtm.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Support Mask
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(support_mask, cmap='binary')
        ax.set_title('Support Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_support_mask.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Valley Accumulation
        fig, ax = plt.subplots(figsize=(10, 8))
        valley_accum_vis = valley_accum.copy()
        valley_accum_vis[support_mask == 0] = 0
        im = ax.imshow(valley_accum_vis, cmap='Blues')
        ax.set_title('Valley Accumulation')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_valley_accumulation.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Valley Mask
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(valley_mask, cmap='Blues')
        ax.set_title('Valley Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_valley_mask.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Valley Skeleton
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(valley_skeleton, cmap='Blues')
        ax.set_title('Valley Skeleton')
        plt.savefig(os.path.join(self.output_dir, 'debug_valley_skeleton.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Ridge Accumulation
        fig, ax = plt.subplots(figsize=(10, 8))
        ridge_accum_vis = ridge_accum.copy()
        ridge_accum_vis[support_mask == 0] = 0
        im = ax.imshow(ridge_accum_vis, cmap='Reds')
        ax.set_title('Ridge Accumulation')
        plt.colorbar(im, ax=ax)
        plt.savefig(os.path.join(self.output_dir, 'debug_ridge_accumulation.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Ridge Mask
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ridge_mask, cmap='Reds')
        ax.set_title('Ridge Mask')
        plt.savefig(os.path.join(self.output_dir, 'debug_ridge_mask.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        # Ridge Skeleton
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ridge_skeleton, cmap='Reds')
        ax.set_title('Ridge Skeleton')
        plt.savefig(os.path.join(self.output_dir, 'debug_ridge_skeleton.png'), dpi=80, bbox_inches='tight')
        plt.close()
        
        print(f"[✓] 保存调试图像（8 张）")

    def _run_single_source(self, return_result: bool = False):
        """执行完整流程"""
        print("\n========== 开始提取山谷和山脊 ==========\n")
        
        # 1. 生成 DTM
        dtm, transform, shape, raw_valid_mask = self.build_dtm()
        
        # 2. 有限距离填补
        resolution = self.config['dtm']['resolution']
        max_fill_distance = self.config['dtm']['max_fill_distance']
        dtm_filled, support_mask = self.fill_nodata_limited(dtm, raw_valid_mask, max_fill_distance, resolution)

        edge_cfg = self.config.get("edge_filter", {})
        edge_dist = None
        ridge_support_mask = support_mask.copy()

        if edge_cfg.get("enabled", True):
            edge_dist = self.compute_edge_distance(support_mask, resolution)
            ridge_edge_buffer = float(edge_cfg.get("ridge_edge_buffer_m", 50.0))

            ridge_core_mask = (support_mask > 0) & (edge_dist >= ridge_edge_buffer)

            ridge_support_mask = support_mask.copy()
            ridge_support_mask[~ridge_core_mask] = 0

            print(
                f"[edge_filter] ridge_core_pixels={int(np.sum(ridge_core_mask))}, "
                f"removed_edge_pixels={int(np.sum((support_mask > 0) & ~ridge_core_mask))}, "
                f"buffer={ridge_edge_buffer}m"
            )
        
        # 3. 高斯平滑（NaN 安全）
        sigma = self.config['dtm']['smooth_sigma_cells']
        dtm_smoothed = self.nan_safe_gaussian_smooth(dtm_filled, support_mask, sigma)
        dtm_for_valley = dtm_smoothed

        ridge_sigma = float(self.config.get("ridge_dtm", {}).get("smooth_sigma_cells", 0.5))
        dtm_for_ridge = self.nan_safe_gaussian_smooth(
            dtm_filled,
            support_mask,
            sigma=ridge_sigma
        )
        
        # 4. 填洼处理
        if self.config.get('hydrology', {}).get('fill_sinks', True):
            dtm_filled_sinks = self.fill_sinks(dtm_for_valley, support_mask)
        else:
            dtm_filled_sinks = dtm_for_valley
        
        # 5. D8 流向
        flow_to_r, flow_to_c = self.compute_flow_direction(dtm_filled_sinks, support_mask, resolution)
        
        # 6. 汇流累积
        accumulation = self.compute_flow_accumulation(dtm_filled_sinks, flow_to_r, flow_to_c, support_mask)
        
        # 7. 反地形流向与汇流（用于山脊主线）
        # Ridge extraction now uses original D8 flow and watershed divides, not inverted terrain flow.
        accumulation_inv = np.zeros_like(accumulation, dtype=np.float32)
        ridge_mask = np.zeros_like(accumulation, dtype=np.uint8)

        method = self.config.get('extraction', {}).get('method', 'skeleton')
        min_cells = self.config.get('extraction', {}).get('min_accumulation_cells', 10)
        ridge_lines = []

        if method == 'flow_trace':
            valley_cfg = self.config.get('valley', {})
            vp = valley_cfg.get('primary', valley_cfg)

            valley_lines = self.trace_main_flow_lines(
                accumulation,
                flow_to_r,
                flow_to_c,
                support_mask,
                transform,
                seed_percentile=vp.get('seed_percentile', 98.0),
                continue_percentile=vp.get('continue_percentile', 90.0),
                min_length=vp.get('min_line_length', 80.0),
                keep_top_n=vp.get('keep_top_n', 30),
                feature_type="valley"
            )

            valley_mask, _ = self.build_accumulation_mask(
                accumulation,
                support_mask,
                percentile=vp.get('continue_percentile', 90.0),
                min_cells=min_cells
            )

        elif method == 'flow_trace_two_stage':
            valley_cfg = self.config.get('valley', {})

            valley_lines = self.extract_two_stage_lines(
                accumulation,
                flow_to_r,
                flow_to_c,
                support_mask,
                transform,
                primary_cfg=valley_cfg['primary'],
                supplement_cfg=valley_cfg['supplement'],
                feature_type="valley"
            )

            valley_primary_cfg = valley_cfg.get('primary', valley_cfg)
            valley_mask, _ = self.build_accumulation_mask(
                accumulation,
                support_mask,
                percentile=valley_primary_cfg.get('continue_percentile', 83.0),
                min_cells=min_cells
            )

            if self.config.get('output', {}).get('save_seed_debug', False):
                self.save_debug_trace_seeds(
                    accumulation, accumulation_inv, support_mask,
                    dtm_filled_sinks, transform
                )

        else:
            # skeleton method
            valley_mask = self.extract_valley_lines(accumulation, support_mask)
            valley_lines = self.skeleton_to_lines(valley_mask, transform, 'valley')

        bv_cfg = self.config.get('broad_valley', {})
        valley_flow_lines = list(valley_lines)
        valley_broad_lines = []

        if bv_cfg.get('enabled', False):
            valley_broad_lines = self.extract_broad_valley_lines(
                dtm_filled_sinks, support_mask, transform,
                existing_valley_lines=valley_flow_lines
            )
            print(f"[broad_valley] extracted {len(valley_broad_lines)} broad valley lines")

        ridge_openness_top_lines = []
        ridge_broad_crest_lines = []
        ridge_watershed_divide_lines = []
        ridge_center_supplement_lines = []
        ridge_gap_connect_lines = []
        ridge_top_score = None
        broad_crest_score = None
        profile_ridge_score = None

        post_valley = self.config.get('postprocess_valley', self.config.get('postprocess', {}))

        v_merge = post_valley.get('merge_distance', 0.0)
        v_simp = post_valley.get('simplify_tolerance', 0.0)
        v_angle = post_valley.get('max_merge_angle_deg', 180.0)
        v_iter = int(post_valley.get('max_merge_iterations', 5))

        if v_merge > 0 or v_simp > 0:
            valley_flow_lines = self.postprocess_lines(
                valley_flow_lines, v_merge, v_simp, v_angle, v_iter
            )
            valley_broad_lines = self.postprocess_lines(
                valley_broad_lines, v_merge, v_simp, v_angle, v_iter
            )

        prune_cfg = self.config.get('line_prune', {})
        if prune_cfg.get('enabled', False):
            v_min_dist = prune_cfg.get('valley_min_distance', 18.0)
            prune_near = prune_cfg.get('near_ratio_threshold', 0.80)

            before_v = len(valley_flow_lines)
            valley_flow_lines = self.prune_dense_lines(valley_flow_lines, v_min_dist, prune_near)
            print(f"[prune-flow] {before_v} -> {len(valley_flow_lines)}")

        if valley_broad_lines:
            bv_prune_cfg = self.config.get('broad_valley_prune', {})
            bv_min_dist = bv_prune_cfg.get('min_distance', 12.0)
            bv_prune_near = bv_prune_cfg.get('near_ratio_threshold', 0.85)
            before_bv = len(valley_broad_lines)
            valley_broad_lines = self.prune_dense_lines(valley_broad_lines, bv_min_dist, bv_prune_near)
            print(f"[prune-broad-valley] {before_bv} -> {len(valley_broad_lines)}")

        if valley_broad_lines and bv_cfg.get('enabled', False):
            suppress_dist = float(bv_cfg.get('min_distance_from_existing', 25.0))
            suppress_near = float(bv_cfg.get('near_ratio_threshold', 0.75))
            before_sup = len(valley_flow_lines)
            valley_flow_lines = self.remove_flow_lines_near_broad_valley(
                valley_flow_lines, valley_broad_lines,
                min_distance=suppress_dist, near_ratio_threshold=suppress_near
            )
            print(f"[suppress-flow-near-broad-valley] {before_sup} -> {len(valley_flow_lines)}")

        valley_all_lines = valley_flow_lines + valley_broad_lines
        major_valley_cfg = self.config.get("major_valley_filter", {})
        valley_major_lines = self.select_important_valley_lines(
            valley_all_lines,
            dtm_filled_sinks,
            accumulation,
            transform,
            support_mask,
            major_valley_cfg,
            mode="valley"
        )
        if major_valley_cfg.get("enabled", False) and not valley_major_lines:
            print("[major_valley_filter] fallback to all valley lines")
            valley_major_lines = list(valley_all_lines)

        use_major_for_ridge = major_valley_cfg.get("use_for_ridge", True)
        valley_ridge_lines = valley_major_lines if use_major_for_ridge else valley_all_lines

        major_ids = {id(line) for line in valley_major_lines}
        major_flow_lines = [line for line in valley_flow_lines if id(line) in major_ids]
        major_broad_lines = [line for line in valley_broad_lines if id(line) in major_ids]

        terrain_active_mask = None
        terrain_cfg = self.config.get("terrain_active", {})
        if terrain_cfg.get("enabled", False):
            terrain_active_mask = self.compute_terrain_active_mask(
                dtm_for_ridge,
                ridge_support_mask,
                resolution,
                terrain_cfg
            )
            print(f"[terrain_active] active_pixels={int(np.sum(terrain_active_mask))}")

        rot_cfg = self.config.get("ridge_openness_top", {})
        if rot_cfg.get("enabled", False):
            ridge_openness_top_lines, ridge_top_score, profile_ridge_score = self.extract_openness_top_ridge_lines(
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                valley_lines=valley_ridge_lines,
                edge_dist=edge_dist,
                terrain_active_mask=terrain_active_mask
            )

        ws_cfg = self.config.get("watershed_divide_ridge", {})
        if ws_cfg.get("enabled", False):
            ridge_watershed_divide_lines = self.extract_watershed_divide_ridge_lines(
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                flow_to_r,
                flow_to_c,
                accumulation,
                major_flow_lines if use_major_for_ridge else valley_flow_lines,
                major_broad_lines if use_major_for_ridge else valley_broad_lines,
                edge_dist=edge_dist
            )

        rcs_cfg = self.config.get("ridge_center_supplement", {})
        if rcs_cfg.get("enabled", False):
            ridge_center_supplement_lines = self.extract_valley_center_supplement_ridge_lines(
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                valley_lines=valley_ridge_lines,
                existing_ridge_lines=ridge_openness_top_lines + ridge_watershed_divide_lines,
                edge_dist=edge_dist
            )

        bcr_cfg = self.config.get("broad_crest_ridge", {})
        if bcr_cfg.get("enabled", False):
            ridge_broad_crest_lines, broad_crest_score = self.extract_broad_crest_ridge_lines(
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                valley_lines=valley_ridge_lines,
                existing_ridge_lines=(
                    ridge_openness_top_lines
                    + ridge_watershed_divide_lines
                    + ridge_center_supplement_lines
                ),
                edge_dist=edge_dist,
                terrain_active_mask=terrain_active_mask
            )

        ridge_score_for_filter = ridge_top_score
        if broad_crest_score is not None:
            if ridge_score_for_filter is None:
                ridge_score_for_filter = broad_crest_score
            else:
                a = ridge_score_for_filter
                b = broad_crest_score
                ridge_score_for_filter = np.where(
                    np.isfinite(a) & np.isfinite(b),
                    np.maximum(a, b),
                    np.where(np.isfinite(a), a, b)
                ).astype(np.float32)

        ridge_lines = (
            ridge_openness_top_lines
            + ridge_broad_crest_lines
            + ridge_watershed_divide_lines
            + ridge_center_supplement_lines
        )
        ridge_candidates_total = len(ridge_lines)

        ridge_after_edge_filter = len(ridge_lines)
        if edge_dist is not None and ridge_lines:
            edge_line_cfg = self.config.get("edge_filter", {})
            ridge_lines = self.filter_lines_by_edge_distance(
                ridge_lines,
                edge_dist,
                transform,
                min_edge_distance_m=float(edge_line_cfg.get("endpoint_edge_buffer_m", 70.0)),
                max_near_edge_ratio=float(edge_line_cfg.get("max_near_edge_ratio", 0.10)),
                n_samples=40,
                reject_if_endpoint_near_edge=bool(edge_line_cfg.get("reject_if_endpoint_near_edge", True)),
                endpoint_edge_buffer_m=float(edge_line_cfg.get("endpoint_edge_buffer_m", 70.0)),
                short_line_length_m=float(edge_line_cfg.get("short_line_length_m", 120.0)),
                short_line_near_edge_ratio=float(edge_line_cfg.get("short_line_near_edge_ratio", 0.05))
            )
            ridge_after_edge_filter = len(ridge_lines)
            print(f"[ridge-edge-filter] {ridge_candidates_total} -> {ridge_after_edge_filter}")

        rgc_cfg = self.config.get("ridge_gap_connect", {})
        if rgc_cfg.get("enabled", False) and ridge_lines and ridge_score_for_filter is not None:
            corridor_score, gap_tpi, gap_valley_dist = self.compute_ridge_corridor_score(
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                valley_ridge_lines,
                ridge_score_for_filter,
                profile_ridge_score,
                edge_dist=edge_dist,
                terrain_active_mask=terrain_active_mask
            )

            current_ridge_lines = list(ridge_lines)
            iterations = max(1, int(rgc_cfg.get("iterations", 1)))
            for _ in range(iterations):
                connectors = self.connect_ridge_gaps_by_cost_path(
                    current_ridge_lines,
                    dtm_for_ridge,
                    ridge_support_mask,
                    transform,
                    ridge_score_for_filter,
                    gap_tpi,
                    gap_valley_dist,
                    rgc_cfg,
                    profile_score=profile_ridge_score,
                    edge_dist=edge_dist,
                    corridor_score=corridor_score
                )
                if not connectors:
                    break
                conn_cfg = self.config.get("connector_final_filter", {})
                if conn_cfg.get("enabled", False):
                    connectors = self.filter_final_ridge_lines_by_profile(
                        connectors,
                        dtm_for_ridge,
                        ridge_support_mask,
                        transform,
                        ridge_score=corridor_score,
                        valley_lines=valley_ridge_lines,
                        edge_dist=edge_dist,
                        cfg=conn_cfg
                    )
                    if not connectors:
                        break
                ridge_gap_connect_lines.extend(connectors)
                current_ridge_lines.extend(connectors)

            if ridge_gap_connect_lines:
                ridge_lines = current_ridge_lines
            print(f"[ridge_gap_connect] total_added={len(ridge_gap_connect_lines)}")

        ridge_after_gap_connect_total = len(ridge_lines)

        post_ridge = self.config.get(
            'postprocess_ridge',
            self.config.get('postprocess', {})
        )
        r_merge = float(post_ridge.get('merge_distance', 0.0))
        r_simp = float(post_ridge.get('simplify_tolerance', 0.0))
        r_angle = float(post_ridge.get('max_merge_angle_deg', 180.0))
        r_iter = int(post_ridge.get('max_merge_iterations', 5))

        if ridge_lines and (r_merge > 0 or r_simp > 0):
            before_r = len(ridge_lines)
            ridge_lines = self.postprocess_lines(
                ridge_lines,
                r_merge,
                r_simp,
                r_angle,
                r_iter
            )
            print(f"[postprocess-ridge] {before_r} -> {len(ridge_lines)}")

        ridge_before_dense_prune = len(ridge_lines)
        ridge_after_dense_prune = len(ridge_lines)
        dense_cfg = self.config.get("ridge_dense_prune", {})
        if dense_cfg.get("enabled", False) and ridge_lines:
            before_dense = len(ridge_lines)
            ridge_lines = self.prune_dense_ridge_lines(
                ridge_lines,
                dtm_for_ridge,
                transform,
                ridge_score_for_filter,
                dense_cfg
            )
            ridge_after_dense_prune = len(ridge_lines)
            print(f"[ridge_dense_prune] removed={before_dense - ridge_after_dense_prune}")

        ridge_before_final_filter = len(ridge_lines)

        rff_cfg = self.config.get("ridge_final_filter", {})
        if rff_cfg.get("enabled", False):
            ridge_lines = self.filter_final_ridge_lines_by_profile(
                ridge_lines,
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                ridge_score=ridge_score_for_filter,
                valley_lines=valley_ridge_lines,
                edge_dist=edge_dist,
                cfg=rff_cfg
            )

        bcf_cfg = self.config.get("broad_crest_final_filter", {})
        if ridge_broad_crest_lines and bcf_cfg:
            broad_filter_cfg = dict(bcf_cfg)
            broad_filter_cfg["enabled"] = True
            broad_filter_cfg.setdefault(
                "profile_half_width_m",
                self.config.get("ridge_final_filter", {}).get("profile_half_width_m", 80.0)
            )
            broad_filter_cfg.setdefault(
                "min_distance_to_valley_m",
                self.config.get("ridge_final_filter", {}).get("min_distance_to_valley_m", 5.0)
            )
            broad_filter_cfg.setdefault(
                "min_edge_distance_m",
                self.config.get("ridge_final_filter", {}).get("min_edge_distance_m", 30.0)
            )
            broad_kept = self.filter_final_ridge_lines_by_profile(
                ridge_broad_crest_lines,
                dtm_for_ridge,
                ridge_support_mask,
                transform,
                ridge_score=ridge_score_for_filter,
                valley_lines=valley_ridge_lines,
                edge_dist=edge_dist,
                cfg=broad_filter_cfg
            )
            broad_extra = self.filter_supplement_lines(
                broad_kept,
                ridge_lines,
                float(self.config.get("broad_crest_ridge", {}).get("min_distance_from_existing_ridge_m", 35.0)),
                0.75,
                0.0,
                0
            )
            if broad_extra:
                ridge_lines = ridge_lines + broad_extra
                print(f"[broad_crest_final_filter] readded={len(broad_extra)}")

        use_major_for_output = (
            major_valley_cfg.get("enabled", False)
            and major_valley_cfg.get("use_for_output", True)
        )
        if use_major_for_output:
            valley_lines = list(valley_major_lines)
        else:
            valley_lines = list(valley_all_lines)

        valley_method_map = {}
        valley_source_by_id = {id(line): "flow_trace" for line in valley_flow_lines}
        valley_source_by_id.update({id(line): "broad_valley" for line in valley_broad_lines})
        for i, line in enumerate(valley_lines):
            valley_method_map[i] = valley_source_by_id.get(id(line), "major_valley")

        ridge_method_map = {}
        for i in range(len(ridge_lines)):
            ridge_method_map[i] = "ridge_openness_top_combined"

        print(
            f"[最终统计] valley_all_total={len(valley_all_lines)}, "
            f"valley_major_total={len(valley_major_lines)}, "
            f"valley_output_total={len(valley_lines)}, "
            f"ridge_candidates_total={ridge_candidates_total}, "
            f"ridge_after_edge_filter={ridge_after_edge_filter}, "
            f"ridge_openness_top={len(ridge_openness_top_lines)}, "
            f"ridge_broad_crest={len(ridge_broad_crest_lines)}, "
            f"ridge_watershed_divide={len(ridge_watershed_divide_lines)}, "
            f"ridge_center_supplement={len(ridge_center_supplement_lines)}, "
            f"ridge_gap_connect={len(ridge_gap_connect_lines)}, "
            f"ridge_after_gap_connect_total={ridge_after_gap_connect_total}, "
            f"ridge_after_dense_prune={ridge_after_dense_prune}, "
            f"ridge_before_final_filter={ridge_before_final_filter}, "
            f"ridge_removed_by_edge={ridge_candidates_total - ridge_after_edge_filter}, "
            f"ridge_removed_by_dense_prune={ridge_before_dense_prune - ridge_after_dense_prune}, "
            f"ridge_removed_by_final_filter={ridge_before_final_filter - len(ridge_lines)}, "
            f"ridge_total={len(ridge_lines)}"
        )

        imp_cfg = self.config.get('line_importance', {})
        valley_importance = None
        ridge_importance = None
        if imp_cfg.get('enabled', False):
            print("[importance] 正在评估山谷线重要性...")
            valley_importance = self.evaluate_lines_importance(
                valley_lines, dtm_filled_sinks, transform,
                accumulation, support_mask, 'valley'
            )
            print("[importance] 正在评估山脊线重要性...")
            ridge_importance = self.evaluate_lines_importance(
                ridge_lines, dtm_for_ridge, transform,
                accumulation_inv, support_mask, 'ridge'
            )
            v_high = sum(1 for v in valley_importance if v['importance_level'] == 'high')
            v_med = sum(1 for v in valley_importance if v['importance_level'] == 'medium')
            v_low = sum(1 for v in valley_importance if v['importance_level'] == 'low')
            r_high = sum(1 for v in ridge_importance if v['importance_level'] == 'high')
            r_med = sum(1 for v in ridge_importance if v['importance_level'] == 'medium')
            r_low = sum(1 for v in ridge_importance if v['importance_level'] == 'low')
            print(f"[importance] valley: high={v_high}, medium={v_med}, low={v_low}")
            print(f"[importance] ridge:  high={r_high}, medium={r_med}, low={r_low}")

        out_cfg = self.config.get('output', {})

        if out_cfg.get('save_debug_images', False):
            ridge_skeleton = morphology.skeletonize(ridge_mask > 0)
            valley_skeleton = morphology.skeletonize(valley_mask > 0)
            self.save_debug_images(dtm_filled_sinks, support_mask, accumulation, valley_mask, valley_skeleton,
                                   accumulation_inv, ridge_mask, ridge_skeleton, transform)

        if out_cfg.get('save_stage_previews', False):
            self.create_preview_single(
                dtm_filled_sinks, transform, support_mask,
                valley_flow_lines, ridge_lines,
                line_color='blue', line_label='Valley (flow_trace)',
                filename='preview_flow_only.png',
                title=f'Flow Trace Valley ({len(valley_flow_lines)} lines)'
            )
            self.create_preview_single(
                dtm_filled_sinks, transform, support_mask,
                valley_broad_lines, ridge_lines,
                line_color='cyan', line_label='Valley (broad)',
                filename='preview_broad_valley_only.png',
                title=f'Broad Valley ({len(valley_broad_lines)} lines)'
            )
            self.create_preview_single(
                dtm_filled_sinks, transform, support_mask,
                ridge_watershed_divide_lines, [],
                line_color='red', line_label='Ridge (watershed divide)',
                filename='preview_ridge_watershed_divide_only.png',
                title=f'Watershed Divide Ridge ({len(ridge_watershed_divide_lines)} lines)'
            )
            self.create_preview_single(
                dtm_filled_sinks, transform, support_mask,
                valley_lines, ridge_lines,
                line_color='blue', line_label='Valley (combined)',
                filename='preview_combined.png',
                title=f'Combined Valley ({len(valley_lines)}) + Ridge ({len(ridge_lines)})'
            )

        self.save_terrain_features_geojson(valley_lines, ridge_lines, valley_method_map, ridge_method_map,
                                             valley_importance, ridge_importance)

        if self.config['output'].get('save_feature_points', True):
            valley_lines_for_points = valley_lines
            ridge_lines_for_points = ridge_lines

            pm_cfg = self.config.get('point_mapping', {})
            if pm_cfg.get('use_importance_filter', False) and imp_cfg.get('enabled', False):
                min_level = pm_cfg.get('min_importance_level', 'medium')
                valley_lines_for_points = self.filter_lines_by_importance(
                    valley_lines, valley_importance, min_level
                )
                ridge_lines_for_points = self.filter_lines_by_importance(
                    ridge_lines, ridge_importance, min_level
                )
                print(
                    f"[mapping-filter] valley {len(valley_lines)} -> {len(valley_lines_for_points)}, "
                    f"ridge {len(ridge_lines)} -> {len(ridge_lines_for_points)}, min_level={min_level}"
                )

            point_type = self.map_lines_to_points(
                valley_lines_for_points, ridge_lines_for_points,
                transform, shape, support_mask
            )
            if point_type:
                self.save_feature_points_las(point_type)
            else:
                print("[!] 警告：没有找到符合条件的特征点")
        else:
            print("[debug] 跳过 terrain_feature_points.las 输出，仅检查 GeoJSON 和 preview")

        if out_cfg.get('save_final_preview', True):
            preview_flow_lines = major_flow_lines if use_major_for_output else valley_flow_lines
            preview_broad_lines = major_broad_lines if use_major_for_output else valley_broad_lines
            self.create_preview(dtm_filled_sinks, transform, support_mask,
                               valley_lines, ridge_lines,
                               flow_lines=preview_flow_lines, broad_lines=preview_broad_lines)

        result = {
            "ridge_lines": ridge_lines,
            "valley_lines": valley_lines,
            "transform": transform,
            "shape": shape,
            "raw_valid_mask": raw_valid_mask,
            "support_mask": support_mask,
            "dtm": dtm,
            "dtm_filled": dtm_filled,
            "dtm_analysis": dtm_filled_sinks,
            "valley_count": len(valley_lines),
            "ridge_count": len(ridge_lines),
        }

        print("\n========== 提取完成 ==========\n")
        print(f"输出文件位置：{os.path.abspath(self.output_dir)}")
        if return_result:
            return result
        return None

    def extract_structure_from_dtm(self, dtm: np.ndarray, transform: Affine,
                                   shape: Tuple[int, int], raw_valid_mask: np.ndarray,
                                   source_name: str):
        """Run the existing extraction pipeline against an externally built DTM."""
        old_output_dir = self.output_dir
        old_config = copy.deepcopy(self.config)
        old_override = self._dtm_override
        try:
            source_dir = os.path.join(old_output_dir, source_name)
            Path(source_dir).mkdir(parents=True, exist_ok=True)
            self.output_dir = source_dir
            self.config["output_dir"] = source_dir
            self._dtm_override = (dtm, transform, shape, raw_valid_mask)
            self.config.setdefault("output", {})["save_feature_points"] = False

            source_cfg = self.config.get("dual_source", {}).get(source_name, {})
            if "smooth_sigma_cells" in source_cfg:
                self.config.setdefault("dtm", {})["smooth_sigma_cells"] = float(source_cfg["smooth_sigma_cells"])

            result = self._run_single_source(return_result=True)
            result["source_name"] = source_name
            result["output_dir"] = source_dir
            prefix = "candidate" if source_name == "candidate_source" else "pass1"
            self._save_source_named_outputs(result, source_dir, prefix)
            return result
        finally:
            self.output_dir = old_output_dir
            self.config = old_config
            self._dtm_override = old_override

    def _write_lines_geojson(self, path: str, lines: List[LineString], feature_type: str):
        features = []
        for line in lines or []:
            if line is None or line.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": list(line.coords),
                },
                "properties": {
                    "feature_type": feature_type,
                },
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    def _save_source_named_outputs(self, result: dict, source_dir: str, prefix: str):
        legacy_features = os.path.join(source_dir, "terrain_features.geojson")
        legacy_preview = os.path.join(source_dir, "preview.png")
        if os.path.exists(legacy_features):
            shutil.copyfile(legacy_features, os.path.join(source_dir, f"{prefix}_terrain_features.geojson"))
        if os.path.exists(legacy_preview):
            shutil.copyfile(legacy_preview, os.path.join(source_dir, f"{prefix}_preview.png"))
        self._write_lines_geojson(
            os.path.join(source_dir, f"{prefix}_ridge_lines.geojson"),
            result.get("ridge_lines", []),
            "ridge"
        )
        self._write_lines_geojson(
            os.path.join(source_dir, f"{prefix}_valley_lines.geojson"),
            result.get("valley_lines", []),
            "valley"
        )

    def buffer_lines_to_mask(self, lines: List[LineString], shape: Tuple[int, int],
                             transform: Affine, buffer_m: float) -> np.ndarray:
        geoms = []
        for line in lines or []:
            if line is not None and not line.is_empty:
                geom = line.buffer(float(buffer_m))
                if not geom.is_empty:
                    geoms.append((geom, 1))
        if not geoms:
            return np.zeros(shape, dtype=np.uint8)
        return rio_features.rasterize(
            shapes=geoms,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

    def _distance_from_mask(self, mask: np.ndarray, resolution: float) -> np.ndarray:
        if int(np.sum(mask > 0)) == 0:
            return np.full(mask.shape, np.inf, dtype=np.float32)
        return ndimage.distance_transform_edt(mask == 0, sampling=resolution).astype(np.float32)

    def compute_conflict_mask(self, candidate_result: dict, pass1_result: dict) -> np.ndarray:
        cfg = self.config.get("structure_fusion", {})
        resolution = float(abs(candidate_result["transform"].a))
        conflict_distance = float(cfg.get("conflict_distance_m", 10.0))
        cand_ridge = candidate_result["ridge_buffer"].astype(bool)
        cand_valley = candidate_result["valley_buffer"].astype(bool)
        pass_ridge = pass1_result["ridge_buffer"].astype(bool)
        pass_valley = pass1_result["valley_buffer"].astype(bool)
        pass_valley_dist = self._distance_from_mask(pass_valley, resolution)
        pass_ridge_dist = self._distance_from_mask(pass_ridge, resolution)
        conflict = (cand_ridge & (pass_valley_dist <= conflict_distance)) | (
            cand_valley & (pass_ridge_dist <= conflict_distance)
        )
        return conflict.astype(np.uint8)

    def fuse_structure_results(self, candidate_result: dict, pass1_result: dict) -> dict:
        shape = candidate_result["shape"]
        transform = candidate_result["transform"]
        resolution = float(abs(transform.a))

        cr = candidate_result["ridge_buffer"].astype(bool)
        cv = candidate_result["valley_buffer"].astype(bool)
        pr = pass1_result["ridge_buffer"].astype(bool)
        pv = pass1_result["valley_buffer"].astype(bool)
        conflict = self.compute_conflict_mask(candidate_result, pass1_result).astype(bool)

        fused_ridge = cr | pr
        fused_valley = cv | pv
        high_ridge = cr & pr
        high_valley = cv & pv
        fused_structure = fused_ridge | fused_valley
        high_structure = high_ridge | high_valley

        candidate_structure = cr | cv
        pass1_structure = pr | pv
        source = np.zeros(shape, dtype=np.uint8)
        source[candidate_structure & ~pass1_structure] = 1
        source[pass1_structure & ~candidate_structure] = 2
        source[candidate_structure & pass1_structure] = 3
        source[conflict] = 4

        confidence = np.zeros(shape, dtype=np.float32)
        confidence[source == 1] = 0.70
        confidence[source == 2] = 0.45
        confidence[source == 3] = 1.00
        confidence[source == 4] = 0.20

        ridge_distance = np.minimum(
            self._distance_from_mask(cr, resolution),
            self._distance_from_mask(pr, resolution)
        ).astype(np.float32)
        valley_distance = np.minimum(
            self._distance_from_mask(cv, resolution),
            self._distance_from_mask(pv, resolution)
        ).astype(np.float32)

        return {
            "fused_ridge_zone": fused_ridge.astype(np.uint8),
            "fused_valley_zone": fused_valley.astype(np.uint8),
            "fused_structure_zone": fused_structure.astype(np.uint8),
            "high_confidence_ridge_zone": high_ridge.astype(np.uint8),
            "high_confidence_valley_zone": high_valley.astype(np.uint8),
            "high_confidence_structure_zone": high_structure.astype(np.uint8),
            "structure_source": source,
            "structure_confidence": confidence,
            "structure_conflict_mask": conflict.astype(np.uint8),
            "ridge_distance": ridge_distance,
            "valley_distance": valley_distance,
            "transform": transform,
            "shape": shape,
        }

    def _rasterio_crs(self):
        if self.crs is None:
            return None
        try:
            return self.crs.to_wkt()
        except Exception:
            return self.crs

    def write_single_band_tif(self, path: str, array: np.ndarray, transform: Affine,
                              dtype: str, nodata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=dtype,
            crs=self._rasterio_crs(),
            transform=transform,
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(array.astype(dtype), 1)
        return path

    def build_simple_openness_masks(self, openness: np.ndarray, tpi: np.ndarray,
                                    support_mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
        valid = (support_mask > 0) & np.isfinite(openness) & np.isfinite(tpi)
        if int(np.sum(valid)) == 0:
            return np.zeros_like(support_mask, dtype=np.uint8), np.zeros_like(support_mask, dtype=np.uint8)

        valley_openness_th = np.percentile(openness[valid], float(cfg.get("valley_openness_percentile", 35.0)))
        ridge_openness_th = np.percentile(openness[valid], float(cfg.get("ridge_openness_percentile", 70.0)))
        valley_tpi_th = np.percentile(tpi[valid], float(cfg.get("valley_tpi_percentile", 35.0)))
        ridge_tpi_th = np.percentile(tpi[valid], float(cfg.get("ridge_tpi_percentile", 65.0)))

        valley_mask = valid & (openness <= valley_openness_th) & (tpi <= valley_tpi_th)
        ridge_mask = valid & (openness >= ridge_openness_th) & (tpi >= ridge_tpi_th)

        close_disk = int(cfg.get("closing_disk", 1))
        if close_disk > 0:
            valley_mask = morphology.binary_closing(valley_mask, morphology.disk(close_disk))
            ridge_mask = morphology.binary_closing(ridge_mask, morphology.disk(close_disk))

        min_area_cells = int(cfg.get("min_area_cells", 20))
        if min_area_cells > 0:
            valley_mask = morphology.remove_small_objects(valley_mask & valid, min_size=min_area_cells)
            ridge_mask = morphology.remove_small_objects(ridge_mask & valid, min_size=min_area_cells)

        print(
            f"[simple_openness] valley_openness_th={valley_openness_th:.3f}, "
            f"ridge_openness_th={ridge_openness_th:.3f}, "
            f"valley_cells={int(np.sum(valley_mask))}, ridge_cells={int(np.sum(ridge_mask))}"
        )
        return valley_mask.astype(np.uint8), ridge_mask.astype(np.uint8)

    def thin_simple_feature_masks(self, valley_mask: np.ndarray, ridge_mask: np.ndarray,
                                  cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
        valley_line = morphology.skeletonize(valley_mask > 0)
        ridge_line = morphology.skeletonize(ridge_mask > 0)

        prune_cells = int(cfg.get("line_min_cells", 8))
        if prune_cells > 0:
            valley_line = morphology.remove_small_objects(valley_line, min_size=prune_cells)
            ridge_line = morphology.remove_small_objects(ridge_line, min_size=prune_cells)

        print(
            f"[simple_openness] thinned valley_cells={int(np.sum(valley_line))}, "
            f"ridge_cells={int(np.sum(ridge_line))}"
        )
        return valley_line.astype(np.uint8), ridge_line.astype(np.uint8)

    def render_openness_gray(self, openness: np.ndarray, support_mask: np.ndarray, cfg: dict) -> np.ndarray:
        valid = (support_mask > 0) & np.isfinite(openness)
        gray = np.zeros(openness.shape, dtype=np.uint8)
        vals = openness[valid]
        if vals.size == 0:
            return gray

        low = np.percentile(vals, float(cfg.get("render_percentile_low", 2.0)))
        high = np.percentile(vals, float(cfg.get("render_percentile_high", 98.0)))
        if high <= low:
            high = low + 1e-6

        stretched = np.clip((openness - low) / (high - low), 0.0, 1.0)
        gamma = float(cfg.get("render_gamma", 1.0))
        if gamma > 0 and abs(gamma - 1.0) > 1e-6:
            stretched = np.power(stretched, gamma)
        if bool(cfg.get("render_invert", False)):
            stretched = 1.0 - stretched

        gray[valid] = np.round(stretched[valid] * 255.0).astype(np.uint8)
        return gray

    def render_line_on_openness(self, openness_gray: np.ndarray, line_mask: np.ndarray,
                                cfg: dict) -> np.ndarray:
        overlay = openness_gray.copy()
        mask = line_mask > 0
        radius = int(cfg.get("overlay_line_radius_cells", 1))
        if radius > 0 and np.any(mask):
            mask = morphology.binary_dilation(mask, morphology.disk(radius))
        overlay[mask] = int(cfg.get("overlay_line_value", 255))
        return overlay.astype(np.uint8)

    def cleanup_simple_output_dir(self):
        out_dir = Path(self.output_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            return

        known_files = {
            "openness.tif",
            "valley_on_openness.tif",
            "ridge_on_openness.tif",
            "class3_openness_features.las",
            "terrain_features.geojson",
            "terrain_feature_points.las",
            "preview.png",
            "preview_flow_only.png",
            "preview_broad_valley_only.png",
            "preview_ridge_watershed_divide_only.png",
            "preview_combined.png",
        }
        known_dirs = {"fused", "candidate_source", "pass1_source"}
        known_prefixes = ("debug_",)

        for child in list(out_dir.iterdir()):
            remove = child.name in known_files or child.name.startswith(known_prefixes)
            if child.is_dir():
                remove = child.name in known_dirs
            if not remove:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def save_simple_class3_points(self, valley_line_mask: np.ndarray, ridge_line_mask: np.ndarray, transform: Affine,
                                  cfg: dict, output_path: str) -> int:
        las = self._ensure_las_loaded()
        cls = np.asarray(las.classification)
        include_classes = set(int(v) for v in cfg.get("classes", self.ground_classes))
        keep_class = np.isin(cls, list(include_classes))
        buffer_distance = float(cfg.get(
            "point_buffer_distance",
            self.config.get("point_mapping", {}).get("point_buffer_distance", 5.0)
        ))
        resolution = abs(transform.a) if transform.a != 0 else float(self.config.get("dtm", {}).get("resolution", 1.0))

        xs = np.asarray(las.x)
        ys = np.asarray(las.y)
        rows, cols = valley_line_mask.shape
        col = ((xs - transform.c) / transform.a).astype(np.int64)
        row = ((ys - transform.f) / transform.e).astype(np.int64)
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)

        selected = np.zeros(len(cls), dtype=bool)
        point_user_data = np.zeros(len(cls), dtype=np.uint8)
        valid_idx = np.where(keep_class & inside)[0]

        mapping_mode = str(cfg.get("point_mapping_mode", "nearest_line_cell")).lower()
        if mapping_mode == "nearest_line_cell":
            points_per_line_cell = max(1, int(cfg.get("points_per_line_cell", 1)))

            def select_nearest_points(line_mask: np.ndarray, label: int):
                line_rc = np.column_stack(np.where(line_mask > 0))
                if len(line_rc) == 0 or len(valid_idx) == 0:
                    return 0

                point_rc = np.column_stack([row[valid_idx], col[valid_idx]]).astype(np.float32)
                point_xy = point_rc * resolution
                tree = cKDTree(point_xy)
                line_xy = line_rc.astype(np.float32) * resolution
                k = min(points_per_line_cell, len(valid_idx))
                dist, nearest = tree.query(line_xy, k=k, distance_upper_bound=buffer_distance)
                if k == 1:
                    dist = dist[:, np.newaxis]
                    nearest = nearest[:, np.newaxis]
                ok = np.isfinite(dist) & (nearest < len(valid_idx))
                if not np.any(ok):
                    return 0

                chosen = np.unique(valid_idx[nearest[ok].astype(np.int64)])
                selected[chosen] = True
                empty = point_user_data[chosen] == 0
                point_user_data[chosen[empty]] = label
                occupied = ~empty
                point_user_data[chosen[occupied]] = np.where(
                    point_user_data[chosen[occupied]] == label,
                    label,
                    3
                ).astype(np.uint8)
                return int(len(chosen))

            valley_count = select_nearest_points(valley_line_mask, 1)
            ridge_count = select_nearest_points(ridge_line_mask, 2)
            print(
                f"[simple_openness] nearest point mapping: "
                f"valley_points={valley_count}, ridge_points={ridge_count}, "
                f"search_radius={buffer_distance}m, points_per_line_cell={points_per_line_cell}"
            )
        else:
            structure_mask = ((valley_line_mask > 0) | (ridge_line_mask > 0))
            if buffer_distance > 0 and np.any(structure_mask):
                max_dist_cells = buffer_distance / resolution
                valley_hit_mask = ndimage.distance_transform_edt(valley_line_mask == 0) <= max_dist_cells
                ridge_hit_mask = ndimage.distance_transform_edt(ridge_line_mask == 0) <= max_dist_cells
            else:
                valley_hit_mask = valley_line_mask > 0
                ridge_hit_mask = ridge_line_mask > 0

            if len(valid_idx):
                valley_hit = valley_hit_mask[row[valid_idx], col[valid_idx]] > 0
                ridge_hit = ridge_hit_mask[row[valid_idx], col[valid_idx]] > 0
                hit = valley_hit | ridge_hit
                selected[valid_idx] = hit
                labels = np.zeros(len(valid_idx), dtype=np.uint8)
                labels[valley_hit] = 1
                labels[ridge_hit] = 2
                labels[valley_hit & ridge_hit] = 3
                point_user_data[valid_idx] = labels

        indices = np.where(selected)[0]
        if len(indices) == 0:
            print("[simple_openness] no points selected for class3 output")
            return 0

        output_las = laspy.LasData(self.make_output_las_header(las))
        output_las.points = las.points[indices].copy()
        output_las.classification = np.full(len(indices), int(cfg.get("output_class", 3)), dtype=np.uint8)
        output_las.user_data = point_user_data[indices].astype(np.uint8)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output_las.write(output_path)
        print(f"[simple_openness] saved class3 LAS: {output_path} ({len(indices)} points)")
        return int(len(indices))

    def run_simple_openness(self):
        print("\n========== simple openness extraction ==========\n")
        cfg = self.config.get("simple_openness", {})
        self.cleanup_simple_output_dir()

        dtm, transform, shape, raw_valid_mask = self.build_dtm()
        resolution = float(self.config["dtm"]["resolution"])
        max_fill_distance = float(self.config["dtm"]["max_fill_distance"])
        dtm_filled, support_mask = self.fill_nodata_limited(dtm, raw_valid_mask, max_fill_distance, resolution)

        valley_sigma = float(self.config.get("dtm", {}).get("smooth_sigma_cells", 1.2))
        ridge_sigma = float(cfg.get("smooth_sigma_cells", self.config.get("ridge_dtm", {}).get("smooth_sigma_cells", 0.9)))
        dtm_for_valley = self.nan_safe_gaussian_smooth(dtm_filled, support_mask, sigma=valley_sigma)
        dtm_for_ridge = self.nan_safe_gaussian_smooth(dtm_filled, support_mask, sigma=ridge_sigma)

        openness = self.compute_positive_openness(
            dtm_for_ridge,
            support_mask,
            resolution,
            radius_m=float(cfg.get("radius_m", 120.0)),
            sample_step_m=float(cfg.get("sample_step_m", 6.0)),
            directions=int(cfg.get("directions", 8))
        )

        openness_out = self.render_openness_gray(openness, support_mask, cfg)

        if self.config.get("hydrology", {}).get("fill_sinks", False):
            dtm_filled_sinks = self.fill_sinks(dtm_for_valley, support_mask)
        else:
            dtm_filled_sinks = dtm_for_valley

        flow_to_r, flow_to_c = self.compute_flow_direction(dtm_filled_sinks, support_mask, resolution)
        accumulation = self.compute_flow_accumulation(dtm_filled_sinks, flow_to_r, flow_to_c, support_mask)

        valley_cfg = self.config.get("valley", {})
        valley_lines = self.extract_two_stage_lines(
            accumulation,
            flow_to_r,
            flow_to_c,
            support_mask,
            transform,
            primary_cfg=valley_cfg.get("primary", valley_cfg),
            supplement_cfg=valley_cfg.get("supplement", {}),
            feature_type="valley"
        )

        post_valley = self.config.get("postprocess_valley", self.config.get("postprocess", {}))
        v_merge = float(post_valley.get("merge_distance", 0.0))
        v_simp = float(post_valley.get("simplify_tolerance", 0.0))
        v_angle = float(post_valley.get("max_merge_angle_deg", 180.0))
        v_iter = int(post_valley.get("max_merge_iterations", 3))
        if valley_lines and (v_merge > 0 or v_simp > 0):
            before_v = len(valley_lines)
            valley_lines = self.postprocess_lines(valley_lines, v_merge, v_simp, v_angle, v_iter)
            print(f"[simple_openness] postprocess valley {before_v} -> {len(valley_lines)}")

        prune_cfg = self.config.get("line_prune", {})
        if prune_cfg.get("enabled", False) and valley_lines:
            before_v = len(valley_lines)
            valley_lines = self.prune_dense_lines(
                valley_lines,
                float(prune_cfg.get("valley_min_distance", 18.0)),
                float(prune_cfg.get("near_ratio_threshold", 0.70))
            )
            print(f"[simple_openness] prune valley {before_v} -> {len(valley_lines)}")

        major_valley_cfg = self.config.get("major_valley_filter", {})
        valley_ridge_lines = valley_lines
        if major_valley_cfg.get("enabled", False) and valley_lines:
            valley_major_lines = self.select_important_valley_lines(
                valley_lines,
                dtm_filled_sinks,
                accumulation,
                transform,
                support_mask,
                major_valley_cfg,
                mode="valley"
            )
            if valley_major_lines and major_valley_cfg.get("use_for_ridge", True):
                valley_ridge_lines = valley_major_lines

        edge_cfg = self.config.get("edge_filter", {})
        edge_dist = None
        ridge_support_mask = support_mask.copy()
        if edge_cfg.get("enabled", True):
            edge_dist = self.compute_edge_distance(support_mask, resolution)
            ridge_edge_buffer = float(edge_cfg.get("ridge_edge_buffer_m", 50.0))
            ridge_core_mask = (support_mask > 0) & (edge_dist >= ridge_edge_buffer)
            ridge_support_mask = support_mask.copy()
            ridge_support_mask[~ridge_core_mask] = 0
            print(
                f"[simple_openness] ridge edge core pixels={int(np.sum(ridge_core_mask))}, "
                f"buffer={ridge_edge_buffer}m"
            )

        terrain_active_mask = None
        terrain_cfg = self.config.get("terrain_active", {})
        if terrain_cfg.get("enabled", False):
            terrain_active_mask = self.compute_terrain_active_mask(
                dtm_for_ridge,
                ridge_support_mask,
                resolution,
                terrain_cfg
            )
            print(f"[simple_openness] terrain active pixels={int(np.sum(terrain_active_mask))}")

        ridge_walk_support = ridge_support_mask.copy().astype(bool)
        if terrain_active_mask is not None:
            ridge_walk_support &= terrain_active_mask.astype(bool)

        ridge_cfg = self.config.get("ridge_openness_walk", {})
        ridge_xy_lines = extract_openness_ridges(
            openness=openness,
            support=ridge_walk_support,
            transform=transform,
            smooth_iters=ridge_cfg["smooth_iters"],
            smooth_k=ridge_cfg["smooth_k"],
            bg_scales=[tuple(s) for s in ridge_cfg["bg_scales"]],
            seed_pct=ridge_cfg["seed_pct"],
            prom_seed=ridge_cfg["prom_seed"],
            prom_continue_pct=ridge_cfg["prom_continue_pct"],
            prom_continue=ridge_cfg["prom_continue"],
            min_mean_prom_pct=ridge_cfg["min_mean_prom_pct"],
            min_mean_prom=ridge_cfg["min_mean_prom"],
            min_length_cells=ridge_cfg["min_length_cells"],
            keep_top_n=ridge_cfg["keep_top_n"],
            prune_spur_cells=ridge_cfg["prune_spur_cells"],
            min_span_cells=ridge_cfg.get("min_span_cells", 0),
            max_loop_cells=ridge_cfg.get("max_loop_cells", 0),
            max_hole_cells=ridge_cfg.get("max_hole_cells", 0),
            edge_len_frac=ridge_cfg.get("edge_len_frac", 1.0),
            edge_margin=ridge_cfg.get("edge_margin", 3),
            min_ridge_spacing=ridge_cfg.get("min_ridge_spacing", 0),
            max_parallel_overlap=ridge_cfg.get("max_parallel_overlap", 0.5)
        )
        ridge_lines = [
            LineString(coords)
            for coords in ridge_xy_lines
            if coords is not None and len(coords) >= 2
        ]
        print(f"[ridge_openness_walk] lines={len(ridge_lines)}")

        if edge_dist is not None and ridge_lines:
            before_r = len(ridge_lines)
            ridge_lines = self.filter_lines_by_edge_distance(
                ridge_lines,
                edge_dist,
                transform,
                min_edge_distance_m=float(edge_cfg.get("endpoint_edge_buffer_m", 70.0)),
                max_near_edge_ratio=float(edge_cfg.get("max_near_edge_ratio", 0.10)),
                n_samples=40,
                reject_if_endpoint_near_edge=bool(edge_cfg.get("reject_if_endpoint_near_edge", True)),
                endpoint_edge_buffer_m=float(edge_cfg.get("endpoint_edge_buffer_m", 70.0)),
                short_line_length_m=float(edge_cfg.get("short_line_length_m", 120.0)),
                short_line_near_edge_ratio=float(edge_cfg.get("short_line_near_edge_ratio", 0.05))
            )
            print(f"[simple_openness] ridge edge filter {before_r} -> {len(ridge_lines)}")

        post_ridge = self.config.get("postprocess_ridge", self.config.get("postprocess", {}))
        r_merge = float(post_ridge.get("merge_distance", 0.0))
        r_simp = float(post_ridge.get("simplify_tolerance", 0.0))
        r_angle = float(post_ridge.get("max_merge_angle_deg", 180.0))
        r_iter = int(post_ridge.get("max_merge_iterations", 3))
        if ridge_lines and (r_merge > 0 or r_simp > 0):
            before_r = len(ridge_lines)
            ridge_lines = self.postprocess_lines(ridge_lines, r_merge, r_simp, r_angle, r_iter)
            print(f"[simple_openness] postprocess ridge {before_r} -> {len(ridge_lines)}")

        valley_line_mask = self.rasterize_lines(valley_lines, shape, transform)
        ridge_line_mask = self.rasterize_lines(ridge_lines, shape, transform)

        self.write_single_band_tif(
            os.path.join(self.output_dir, "openness.tif"),
            openness_out,
            transform,
            "uint8",
            0
        )
        self.write_single_band_tif(
            os.path.join(self.output_dir, "valley_on_openness.tif"),
            self.render_line_on_openness(openness_out, valley_line_mask, cfg),
            transform,
            "uint8",
            0
        )
        self.write_single_band_tif(
            os.path.join(self.output_dir, "ridge_on_openness.tif"),
            self.render_line_on_openness(openness_out, ridge_line_mask, cfg),
            transform,
            "uint8",
            0
        )
        class3_count = self.save_simple_class3_points(
            valley_line_mask,
            ridge_line_mask,
            transform,
            cfg,
            os.path.join(self.output_dir, "class3_openness_features.las")
        )
        print(
            f"[simple_openness] summary: valley_line_cells={int(np.sum(valley_line_mask))}, "
            f"ridge_line_cells={int(np.sum(ridge_line_mask))}, class3_points={class3_count}"
        )
        print(f"[simple_openness] output={os.path.abspath(self.output_dir)}")

    def save_structure_masks(self, fused: dict, candidate_result: dict,
                             pass1_result: dict, unified_grid: dict):
        out_dir = os.path.join(self.output_dir, "fused")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        transform = fused["transform"]
        shape = fused["shape"]
        crs = self._rasterio_crs()

        def write_tif(name, array, dtype, nodata):
            path = os.path.join(out_dir, name)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=shape[0],
                width=shape[1],
                count=1,
                dtype=dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
                compress="deflate",
            ) as dst:
                dst.write(array.astype(dtype), 1)
            return path

        mask_names = [
            "fused_ridge_zone",
            "fused_valley_zone",
            "fused_structure_zone",
            "high_confidence_ridge_zone",
            "high_confidence_valley_zone",
            "high_confidence_structure_zone",
            "structure_source",
            "structure_conflict_mask",
        ]
        for name in mask_names:
            write_tif(f"{name}.tif", fused[name], "uint8", 0)
        write_tif("structure_confidence.tif", fused["structure_confidence"], "float32", 0.0)

        fusion_cfg = self.config.get("structure_fusion", {})
        if fusion_cfg.get("output_distance_maps", True):
            rd = np.where(np.isfinite(fused["ridge_distance"]), fused["ridge_distance"], -9999.0)
            vd = np.where(np.isfinite(fused["valley_distance"]), fused["valley_distance"], -9999.0)
            write_tif("ridge_distance.tif", rd, "float32", -9999.0)
            write_tif("valley_distance.tif", vd, "float32", -9999.0)

        cand_usage = dict(self.source_class_usage.get("candidate_source", {}))
        class1_mask = cand_usage.pop("candidate_class1_used_mask", None)
        if class1_mask is not None:
            candidate_dir = os.path.join(self.output_dir, "candidate_source")
            Path(candidate_dir).mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                os.path.join(candidate_dir, "candidate_class1_used_mask.tif"),
                "w",
                driver="GTiff",
                height=shape[0],
                width=shape[1],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
                nodata=0,
                compress="deflate",
            ) as dst:
                dst.write(class1_mask.astype(np.uint8), 1)
            with open(os.path.join(candidate_dir, "candidate_class_usage.json"), "w", encoding="utf-8") as f:
                json.dump(cand_usage, f, indent=2)

        cell_area = float(abs(transform.a * transform.e))
        summary = {
            "input_las": self.input_las,
            "candidate_classes": self.config.get("dual_source", {}).get("candidate_source", {}).get("classes", [2, 1]),
            "pass1_classes": self.config.get("dual_source", {}).get("pass1_source", {}).get("classes", [2, 16]),
            "excluded_classes": [4],
            "unified_grid_shape": [int(shape[0]), int(shape[1])],
            "unified_resolution": float(abs(transform.a)),
            "candidate_ridge_count": int(candidate_result.get("ridge_count", 0)),
            "candidate_valley_count": int(candidate_result.get("valley_count", 0)),
            "pass1_ridge_count": int(pass1_result.get("ridge_count", 0)),
            "pass1_valley_count": int(pass1_result.get("valley_count", 0)),
            "fused_ridge_area_m2": float(np.sum(fused["fused_ridge_zone"] > 0) * cell_area),
            "fused_valley_area_m2": float(np.sum(fused["fused_valley_zone"] > 0) * cell_area),
            "high_confidence_area_m2": float(np.sum(fused["high_confidence_structure_zone"] > 0) * cell_area),
            "conflict_area_m2": float(np.sum(fused["structure_conflict_mask"] > 0) * cell_area),
        }
        summary.update(cand_usage)
        with open(os.path.join(out_dir, "structure_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if fusion_cfg.get("output_debug_preview", True):
            self.create_fused_structure_preview(fused, os.path.join(out_dir, "fused_structure_preview.png"))

        if fusion_cfg.get("output_point_las", True):
            self.save_fused_structure_points_las(fused, out_dir)

    def save_fused_structure_points_las(self, fused: dict, out_dir: str):
        """
        Save original LAS points that fall inside fused ridge/valley zones.
        user_data: 1=valley, 2=ridge, 3=ridge+valley overlap, 4=conflict.
        classification is preserved.
        """
        las = self._ensure_las_loaded()
        transform = fused["transform"]
        rows, cols = fused["shape"]
        cfg = self.config.get("structure_fusion", {})
        include_classes_cfg = cfg.get("point_include_classes")
        include_classes = (
            set(int(v) for v in include_classes_cfg)
            if include_classes_cfg is not None
            else None
        )
        exclude_classes = set(int(v) for v in cfg.get("point_exclude_classes", [4]))

        cls = np.asarray(las.classification)
        keep_class = np.ones(len(cls), dtype=bool)
        if include_classes is not None:
            keep_class &= np.isin(cls, list(include_classes))
        if exclude_classes:
            keep_class &= ~np.isin(cls, list(exclude_classes))
        xs = np.asarray(las.x)
        ys = np.asarray(las.y)

        col = ((xs - transform.c) / transform.a).astype(np.int64)
        row = ((ys - transform.f) / transform.e).astype(np.int64)
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)

        valid = keep_class & inside
        labels = np.zeros(len(xs), dtype=np.uint8)
        rr = row[valid]
        cc = col[valid]
        valid_idx = np.where(valid)[0]

        ridge_hit = fused["fused_ridge_zone"][rr, cc] > 0
        valley_hit = fused["fused_valley_zone"][rr, cc] > 0
        conflict_hit = fused["structure_conflict_mask"][rr, cc] > 0

        valid_labels = np.zeros(len(valid_idx), dtype=np.uint8)
        valid_labels[valley_hit] = 1
        valid_labels[ridge_hit] = 2
        valid_labels[ridge_hit & valley_hit] = 3
        valid_labels[conflict_hit] = 4
        labels[valid_idx] = valid_labels

        def write_subset(filename: str, select_mask: np.ndarray, code_override: Optional[int] = None):
            indices = np.where(select_mask)[0]
            if len(indices) == 0:
                print(f"[structure-las] skip empty {filename}")
                return 0
            output_las = laspy.LasData(self.make_output_las_header(las))
            output_las.points = las.points[indices].copy()
            if code_override is None:
                output_las.user_data = labels[indices].astype(np.uint8)
            else:
                output_las.user_data = np.full(len(indices), int(code_override), dtype=np.uint8)
            output_path = os.path.join(out_dir, filename)
            output_las.write(output_path)
            print(f"[structure-las] saved {filename}: {len(indices)} points")
            return int(len(indices))

        structure_mask = labels > 0
        ridge_mask = labels == 2
        valley_mask = labels == 1
        both_mask = labels == 3
        conflict_mask = labels == 4

        counts = {
            "structure_point_count": write_subset("fused_structure_points.las", structure_mask),
            "ridge_point_count": write_subset("fused_ridge_points.las", ridge_mask | both_mask, code_override=2),
            "valley_point_count": write_subset("fused_valley_points.las", valley_mask | both_mask, code_override=1),
            "conflict_point_count": write_subset("fused_conflict_points.las", conflict_mask, code_override=4),
        }
        with open(os.path.join(out_dir, "structure_point_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "input_las": self.input_las,
                "user_data_codes": {
                    "1": "valley",
                    "2": "ridge",
                    "3": "ridge_and_valley_overlap",
                    "4": "conflict",
                },
                "excluded_classes": sorted(exclude_classes),
                "included_classes": sorted(include_classes) if include_classes is not None else None,
                **counts,
            }, f, indent=2)

    def create_fused_structure_preview(self, fused: dict, output_path: str):
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        src = fused["structure_source"]
        ax.imshow(src, cmap="tab10", vmin=0, vmax=9)
        ax.set_title("Fused Structure Source")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()

    def run_dual_source(self):
        print("\n========== dual source terrain structure extraction ==========\n")
        dual_cfg = self.config.get("dual_source", {})
        fusion_cfg = self.config.get("structure_fusion", {})
        las = self._ensure_las_loaded()
        unified_grid = self.get_unified_grid_bounds(las, exclude_classes=[4])

        candidate_cfg = dual_cfg.get("candidate_source", {})
        pass1_cfg = dual_cfg.get("pass1_source", {})
        candidate_points = self.read_points_by_classes(
            candidate_cfg.get("classes", [2, 1]),
            candidate_cfg.get("exclude_classes", [4, 16])
        )
        pass1_points = self.read_points_by_classes(
            pass1_cfg.get("classes", [2, 16]),
            pass1_cfg.get("exclude_classes", [1, 4])
        )

        candidate_dtm, transform, shape, candidate_raw = self.build_dtm_from_points(
            candidate_points, "candidate_source", unified_grid
        )
        pass1_dtm, pass1_transform, pass1_shape, pass1_raw = self.build_dtm_from_points(
            pass1_points, "pass1_source", unified_grid
        )
        if shape != pass1_shape or transform != pass1_transform:
            raise RuntimeError("dual source grids are not aligned")

        candidate_result = self.extract_structure_from_dtm(
            candidate_dtm, transform, shape, candidate_raw, "candidate_source"
        )
        pass1_result = self.extract_structure_from_dtm(
            pass1_dtm, transform, shape, pass1_raw, "pass1_source"
        )

        ridge_buffer = float(fusion_cfg.get("ridge_buffer_m", 15.0))
        valley_buffer = float(fusion_cfg.get("valley_buffer_m", 18.0))
        for result in (candidate_result, pass1_result):
            result["ridge_buffer"] = self.buffer_lines_to_mask(
                result.get("ridge_lines", []), shape, transform, ridge_buffer
            )
            result["valley_buffer"] = self.buffer_lines_to_mask(
                result.get("valley_lines", []), shape, transform, valley_buffer
            )

        fused = self.fuse_structure_results(candidate_result, pass1_result)
        self.save_structure_masks(fused, candidate_result, pass1_result, unified_grid)
        print(f"[dual_source] output={os.path.abspath(self.output_dir)}")

    def run(self):
        if self.config.get("simple_openness", {}).get("enabled", False):
            return self.run_simple_openness()
        if self.config.get("dual_source", {}).get("enabled", False):
            return self.run_dual_source()
        return self._run_single_source(return_result=False)


def load_config_with_optional_override(config_path: Optional[str]) -> dict:
    config = load_default_config()

    if config_path:
        if not os.path.exists(config_path):
            print(f"[错误] 配置文件不存在：{config_path}")
            sys.exit(1)
        with open(config_path, 'r', encoding='utf-8') as f:
            override = yaml.safe_load(f) or {}
        deep_update(config, override)

    return config


def find_point_cloud_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {'.las', '.laz'}:
            raise ValueError(f"输入文件不是 .las/.laz 点云：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    files = []
    for path in input_path.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.las', '.laz'}:
            continue
        rel_parts = [part.lower() for part in path.relative_to(input_path).parts[:-1]]
        if 'output' in rel_parts or 'openness_output' in rel_parts:
            continue
        files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def output_dir_for_point_cloud(input_las: Path) -> Path:
    return input_las.parent / 'output' / input_las.stem


def simple_output_dir_for_point_cloud(input_las: Path) -> Path:
    return input_las.parent / 'openness_output' / input_las.stem


def main():
    parser = argparse.ArgumentParser(description='提取山谷和山脊线')
    parser.add_argument('--input', default=None, help='输入 .las/.laz 文件或包含点云的文件夹')
    parser.add_argument('--config', default=None, help='可选配置覆盖文件；默认使用 main.py 内置配置')

    args = parser.parse_args()
    config = load_config_with_optional_override(args.config)

    input_arg = args.input or config.get('input_las')
    if not input_arg:
        print("[错误] 请通过 --input 指定 .las/.laz 文件或文件夹")
        sys.exit(1)

    input_path = Path(input_arg).expanduser().resolve()
    try:
        point_cloud_files = find_point_cloud_files(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}")
        sys.exit(1)

    if not point_cloud_files:
        print(f"[错误] 输入文件夹中未找到 .las/.laz 点云：{input_path}")
        sys.exit(1)

    print(f"[batch] 待处理点云数量：{len(point_cloud_files)}")
    for idx, input_las in enumerate(point_cloud_files, start=1):
        output_dir = (
            simple_output_dir_for_point_cloud(input_las)
            if config.get("simple_openness", {}).get("enabled", False)
            else output_dir_for_point_cloud(input_las)
        )
        print(f"\n[batch] ({idx}/{len(point_cloud_files)}) input={input_las}")
        print(f"[batch] output={output_dir}")
        analyzer = TerrainAnalyzer(config, str(input_las), str(output_dir))
        analyzer.run()


if __name__ == '__main__':
    main()
