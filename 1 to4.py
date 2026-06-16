# batch_completed_original_class1_to4_safe.py
# 功能：
# 对比“全分类点云文件”和“TIN滤波后补全点云文件”
# 只将补全点云中：
#   1）坐标对应全分类原始类别1
#   2）并且补全后当前仍然是类别1
# 的点改成类别4
#
# 其他类别，包括类别2地面点、类别5高植被、类别6建筑物，全部保持不变

from pathlib import Path
import csv
import traceback

import numpy as np
import laspy


# ============================================================
# 1. 路径设置
# ============================================================

# 全分类点云文件夹
ORIGINAL_DIR = Path(r"D:\xishudimiandian\shanjixiantiqu\quanfenlei")

# TIN滤波后 + 补全后的点云文件夹
# 这里一定要改成你真实的“补全后文件夹”
COMPLETED_DIR = Path(r"D:\xishudimiandian\dem_repair_project(1)\outputs\predictions\test")

# 输出文件夹
OUTPUT_DIR = Path(r"D:\xishudimiandian\shanjixiantiqu\quanfenlei\1 to4")


# ============================================================
# 2. 类别参数
# ============================================================

# 全分类文件中，原来就是未分类/噪声的类别
ORIGINAL_UNCLASSIFIED_CLASS = 1

# 补全后文件中，只允许当前类别为1的点被改
COMPLETED_CLASS_CAN_CHANGE = 1

# 要改成的新类别
TARGET_CLASS = 4


# ============================================================
# 3. 坐标匹配参数
# ============================================================

# 坐标量化精度，单位和点云坐标单位一致
# 一般点云单位是米，0.001 表示毫米级匹配
# 如果匹配数量明显偏少，可以改成 0.01
COORD_RESOLUTION = 0.001

# 支持的点云格式
POINT_EXTENSIONS = {".las", ".laz"}

# 每批处理多少个类别1点，防止内存过大
CHUNK_SIZE = 1_000_000


def count_class(class_array, class_id):
    return int(np.sum(class_array == class_id))


def class_count_dict(class_array):
    values, counts = np.unique(class_array, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def make_coord_keys(x, y, z, resolution=0.001):
    """
    将实际坐标 x/y/z 转成可比较的坐标 key。
    使用实际坐标，而不是 LAS 原始整数坐标，
    这样即使两个文件的 scale/offset 不完全一致，也能匹配。
    """
    n = len(x)

    coords = np.empty((n, 3), dtype=np.int64)
    coords[:, 0] = np.rint(x / resolution).astype(np.int64)
    coords[:, 1] = np.rint(y / resolution).astype(np.int64)
    coords[:, 2] = np.rint(z / resolution).astype(np.int64)

    coords = np.ascontiguousarray(coords)

    key_dtype = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    keys = coords.view(key_dtype).ravel()

    return keys


def find_original_file(completed_path):
    """
    根据补全后文件名，在全分类文件夹中找对应文件。
    优先完全同名，其次匹配同 stem 的 .las / .laz。
    """
    exact_path = ORIGINAL_DIR / completed_path.name
    if exact_path.exists():
        return exact_path

    for ext in POINT_EXTENSIONS:
        alt_path = ORIGINAL_DIR / f"{completed_path.stem}{ext}"
        if alt_path.exists():
            return alt_path

    return None


def check_output_not_same_as_input(completed_path, output_path):
    """
    防止输出路径和输入路径相同，避免覆盖原始补全结果。
    """
    try:
        if completed_path.resolve() == output_path.resolve():
            raise ValueError(
                "输出文件路径和输入补全文件路径相同，程序停止，避免覆盖原始文件。"
            )
    except FileNotFoundError:
        pass


def process_one_file(original_path, completed_path, output_path):
    print("\n--------------------------------------------------")
    print(f"正在处理：{completed_path.name}")
    print(f"全分类文件：{original_path}")
    print(f"补全后文件：{completed_path}")

    check_output_not_same_as_input(completed_path, output_path)

    original_las = laspy.read(original_path)
    completed_las = laspy.read(completed_path)

    original_classes = np.asarray(original_las.classification)
    completed_classes_before = np.asarray(completed_las.classification).copy()
    completed_classes_after = completed_classes_before.copy()

    original_point_count = len(original_las.points)
    completed_point_count = len(completed_las.points)

    print(f"全分类点数：{original_point_count}")
    print(f"补全后点数：{completed_point_count}")

    # ------------------------------------------------------------
    # 1. 找出全分类文件中原来就是类别1的点
    # ------------------------------------------------------------
    original_class1_mask = original_classes == ORIGINAL_UNCLASSIFIED_CLASS
    original_class1_count = int(np.sum(original_class1_mask))

    print(f"全分类文件中原始类别{ORIGINAL_UNCLASSIFIED_CLASS}点数量：{original_class1_count}")

    completed_counts_before = class_count_dict(completed_classes_before)

    completed_class1_before = completed_counts_before.get(1, 0)
    completed_class2_before = completed_counts_before.get(2, 0)
    completed_class4_before = completed_counts_before.get(4, 0)
    completed_class5_before = completed_counts_before.get(5, 0)
    completed_class6_before = completed_counts_before.get(6, 0)

    if original_class1_count == 0:
        print("该文件中没有原始类别1点，直接复制补全后文件。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed_las.write(output_path)

        completed_counts_after = class_count_dict(completed_classes_after)

        return {
            "file": completed_path.name,
            "original_points": original_point_count,
            "completed_points": completed_point_count,
            "original_class1_count": 0,
            "original_class1_unique_coord_count": 0,
            "changed_to4_count": 0,
            "completed_class1_before": completed_class1_before,
            "completed_class1_after": completed_counts_after.get(1, 0),
            "completed_class2_before": completed_class2_before,
            "completed_class2_after": completed_counts_after.get(2, 0),
            "completed_class4_before": completed_class4_before,
            "completed_class4_after": completed_counts_after.get(4, 0),
            "completed_class5_before": completed_class5_before,
            "completed_class5_after": completed_counts_after.get(5, 0),
            "completed_class6_before": completed_class6_before,
            "completed_class6_after": completed_counts_after.get(6, 0),
            "other_class_changed": "no",
            "output_path": str(output_path),
            "status": "success",
            "error": "",
        }

    # ------------------------------------------------------------
    # 2. 只把全分类文件中的原始类别1点坐标做成 key
    # ------------------------------------------------------------
    original_class1_keys = make_coord_keys(
        original_las.x[original_class1_mask],
        original_las.y[original_class1_mask],
        original_las.z[original_class1_mask],
        resolution=COORD_RESOLUTION,
    )

    original_class1_keys_unique = np.unique(original_class1_keys)
    original_class1_unique_count = len(original_class1_keys_unique)

    print(f"全分类原始类别1唯一坐标数量：{original_class1_unique_count}")

    # ------------------------------------------------------------
    # 3. 只检查补全后当前仍然是类别1的点
    #    这样类别2、类别5、类别6等不会被误改
    # ------------------------------------------------------------
    completed_class1_indices = np.flatnonzero(
        completed_classes_before == COMPLETED_CLASS_CAN_CHANGE
    )

    print(f"补全后当前类别{COMPLETED_CLASS_CAN_CHANGE}点数量：{len(completed_class1_indices)}")

    changed_to4_count = 0

    for start in range(0, len(completed_class1_indices), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(completed_class1_indices))
        idx_chunk = completed_class1_indices[start:end]

        completed_chunk_keys = make_coord_keys(
            completed_las.x[idx_chunk],
            completed_las.y[idx_chunk],
            completed_las.z[idx_chunk],
            resolution=COORD_RESOLUTION,
        )

        # 在补全后的类别1点中，找出坐标对应全分类原始类别1的点
        local_match_mask = np.isin(
            completed_chunk_keys,
            original_class1_keys_unique,
            assume_unique=False,
        )

        matched_indices = idx_chunk[local_match_mask]

        # 核心修改：
        # 只修改补全后当前就是类别1的点
        completed_classes_after[matched_indices] = TARGET_CLASS

        changed_to4_count += int(np.sum(local_match_mask))

    # ------------------------------------------------------------
    # 4. 写回 classification 字段，其他所有属性不变
    # ------------------------------------------------------------
    completed_las.classification = completed_classes_after

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_las.write(output_path)

    # ------------------------------------------------------------
    # 5. 统计检查
    # ------------------------------------------------------------
    completed_counts_after = class_count_dict(completed_classes_after)

    completed_class1_after = completed_counts_after.get(1, 0)
    completed_class2_after = completed_counts_after.get(2, 0)
    completed_class4_after = completed_counts_after.get(4, 0)
    completed_class5_after = completed_counts_after.get(5, 0)
    completed_class6_after = completed_counts_after.get(6, 0)

    # 检查除类别1和类别4外，其他类别是否发生变化
    all_classes = sorted(set(completed_counts_before.keys()) | set(completed_counts_after.keys()))
    changed_other_classes = []

    for cls in all_classes:
        if cls in {COMPLETED_CLASS_CAN_CHANGE, TARGET_CLASS}:
            continue

        before_count = completed_counts_before.get(cls, 0)
        after_count = completed_counts_after.get(cls, 0)

        if before_count != after_count:
            changed_other_classes.append((cls, before_count, after_count))

    other_class_changed = "yes" if changed_other_classes else "no"

    print("处理统计：")
    print(f"  全分类原始类别1点数量：{original_class1_count}")
    print(f"  全分类原始类别1唯一坐标数量：{original_class1_unique_count}")
    print(f"  实际改为类别{TARGET_CLASS}的点数量：{changed_to4_count}")

    print("类别数量变化：")
    print(f"  类别1：{completed_class1_before} -> {completed_class1_after}")
    print(f"  类别2：{completed_class2_before} -> {completed_class2_after}")
    print(f"  类别4：{completed_class4_before} -> {completed_class4_after}")
    print(f"  类别5：{completed_class5_before} -> {completed_class5_after}")
    print(f"  类别6：{completed_class6_before} -> {completed_class6_after}")

    if changed_to4_count != completed_class1_before - completed_class1_after:
        print("警告：类别1减少数量和改成类别4数量不一致，请检查。")

    if changed_to4_count != completed_class4_after - completed_class4_before:
        print("警告：类别4增加数量和改成类别4数量不一致，请检查。")

    if changed_other_classes:
        print("警告：除类别1和类别4之外，还有其他类别数量发生变化：")
        for cls, before_count, after_count in changed_other_classes:
            print(f"  类别{cls}：{before_count} -> {after_count}")
    else:
        print("检查结果：除类别1和类别4外，其他类别数量未变化。")

    print(f"输出文件：{output_path}")

    return {
        "file": completed_path.name,
        "original_points": original_point_count,
        "completed_points": completed_point_count,
        "original_class1_count": original_class1_count,
        "original_class1_unique_coord_count": original_class1_unique_count,
        "changed_to4_count": changed_to4_count,
        "completed_class1_before": completed_class1_before,
        "completed_class1_after": completed_class1_after,
        "completed_class2_before": completed_class2_before,
        "completed_class2_after": completed_class2_after,
        "completed_class4_before": completed_class4_before,
        "completed_class4_after": completed_class4_after,
        "completed_class5_before": completed_class5_before,
        "completed_class5_after": completed_class5_after,
        "completed_class6_before": completed_class6_before,
        "completed_class6_after": completed_class6_after,
        "other_class_changed": other_class_changed,
        "output_path": str(output_path),
        "status": "success",
        "error": "",
    }


def main():
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"全分类文件夹不存在：{ORIGINAL_DIR}")

    if not COMPLETED_DIR.exists():
        raise FileNotFoundError(f"补全后文件夹不存在：{COMPLETED_DIR}")

    if OUTPUT_DIR.resolve() == COMPLETED_DIR.resolve():
        raise ValueError("输出文件夹不能和补全后文件夹相同，避免覆盖原文件。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    completed_files = [
        p for p in COMPLETED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in POINT_EXTENSIONS
    ]

    if not completed_files:
        raise FileNotFoundError(f"补全后文件夹中没有找到 .las 或 .laz 文件：{COMPLETED_DIR}")

    print("开始批量处理")
    print(f"全分类文件夹：{ORIGINAL_DIR}")
    print(f"补全后文件夹：{COMPLETED_DIR}")
    print(f"输出文件夹：{OUTPUT_DIR}")
    print(f"找到补全后点云文件数量：{len(completed_files)}")

    summary_rows = []
    success_count = 0
    skip_count = 0
    fail_count = 0

    for completed_path in completed_files:
        original_path = find_original_file(completed_path)

        if original_path is None:
            print("\n--------------------------------------------------")
            print(f"跳过：找不到对应的全分类文件：{completed_path.name}")

            skip_count += 1

            summary_rows.append({
                "file": completed_path.name,
                "original_points": "",
                "completed_points": "",
                "original_class1_count": "",
                "original_class1_unique_coord_count": "",
                "changed_to4_count": "",
                "completed_class1_before": "",
                "completed_class1_after": "",
                "completed_class2_before": "",
                "completed_class2_after": "",
                "completed_class4_before": "",
                "completed_class4_after": "",
                "completed_class5_before": "",
                "completed_class5_after": "",
                "completed_class6_before": "",
                "completed_class6_after": "",
                "other_class_changed": "",
                "output_path": "",
                "status": "skipped",
                "error": "找不到对应全分类文件",
            })

            continue

        output_path = OUTPUT_DIR / completed_path.name

        try:
            row = process_one_file(original_path, completed_path, output_path)
            summary_rows.append(row)
            success_count += 1

        except Exception as e:
            print("\n处理失败：", completed_path.name)
            print("错误信息：", e)
            traceback.print_exc()

            fail_count += 1

            summary_rows.append({
                "file": completed_path.name,
                "original_points": "",
                "completed_points": "",
                "original_class1_count": "",
                "original_class1_unique_coord_count": "",
                "changed_to4_count": "",
                "completed_class1_before": "",
                "completed_class1_after": "",
                "completed_class2_before": "",
                "completed_class2_after": "",
                "completed_class4_before": "",
                "completed_class4_after": "",
                "completed_class5_before": "",
                "completed_class5_after": "",
                "completed_class6_before": "",
                "completed_class6_after": "",
                "other_class_changed": "",
                "output_path": "",
                "status": "failed",
                "error": str(e),
            })

    summary_path = OUTPUT_DIR / "_completed_original_class1_to4_safe_summary.csv"

    fieldnames = [
        "file",
        "original_points",
        "completed_points",
        "original_class1_count",
        "original_class1_unique_coord_count",
        "changed_to4_count",
        "completed_class1_before",
        "completed_class1_after",
        "completed_class2_before",
        "completed_class2_after",
        "completed_class4_before",
        "completed_class4_after",
        "completed_class5_before",
        "completed_class5_after",
        "completed_class6_before",
        "completed_class6_after",
        "other_class_changed",
        "output_path",
        "status",
        "error",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n==============================")
    print("批量处理完成")
    print(f"成功处理：{success_count}")
    print(f"跳过文件：{skip_count}")
    print(f"失败文件：{fail_count}")
    print(f"输出文件夹：{OUTPUT_DIR}")
    print(f"统计表：{summary_path}")
    print("==============================")


if __name__ == "__main__":
    main()