import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Extract only mean rows from summary.csv")
    parser.add_argument(
        "--input_csv",
        default="/data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/sample_results/scan/evaluation_results/summary.csv",
        help="Path to source summary.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="/data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/sample_results/scan/evaluation_results/summary_means.csv",
        help="Path to output CSV containing only mean rows",
    )
    parser.add_argument(
        "--output_txt",
        default="/data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/sample_results/scan/evaluation_results/summary_means.txt",
        help="Path to aligned text output (space-separated columns)",
    )
    return parser.parse_args()


def is_mean_row(category_value: str) -> bool:
    if category_value is None:
        return False
    value = category_value.strip()
    return value in {"【均值】", "均值", "[均值]", "mean", "Mean"}


def main():
    args = parse_args()

    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    with open(args.input_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if is_mean_row(row.get("类别"))]

    # 导出时去掉“均值”字样，保留方法和指标数值
    for row in rows:
        if "类别" in row:
            row["类别"] = ""

    if not fieldnames:
        raise ValueError("Input CSV has no header.")

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 额外导出一个空格对齐版文本，便于直接阅读
    text_columns = ["方法", "mean_TOPIQ", "PDV", "mean_LPIPS"]
    for col in text_columns:
        if col not in fieldnames:
            raise ValueError(f"Missing expected column in input CSV: {col}")

    widths = {}
    for col in text_columns:
        widths[col] = max(len(col), *(len(str(r.get(col, ""))) for r in rows)) if rows else len(col)

    txt_dir = os.path.dirname(os.path.abspath(args.output_txt))
    if txt_dir:
        os.makedirs(txt_dir, exist_ok=True)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        header = "  ".join(col.ljust(widths[col]) for col in text_columns)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in rows:
            line = "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in text_columns)
            f.write(line + "\n")

    print(f"Done. Kept {len(rows)} mean rows.")
    print(f"Saved to: {args.output_csv}")
    print(f"Saved to: {args.output_txt}")


if __name__ == "__main__":
    main()
