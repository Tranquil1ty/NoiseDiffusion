"""
NoiseDiffusion 参数扫描脚本

扫描 noisediffusion 的 gamma 和 clip(coef) 参数组合，同时计算 noise 和 slerp 基线。
生成完成后自动调用 evaluate.py 计算 TOPIQ / PDV。

参数网格:
  gamma = [0, 0.1, sqrt(0.1), sqrt(0.2)]
  clip  = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, no_clip]

默认类别: aircraft, car, tree, spider, pumpkin, lion_tiger, cup

用法:
  # 单 GPU
  python /data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/scan_params.py --gpus 0

  # 多 GPU 并行（类别轮流分配到各 GPU）
  python /data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/scan_params.py --gpus 0 1 2 3 5

  # 自定义类别
  python /data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/scan_params.py --gpus 0 1 --categories aircraft car

  # 跳过已有结果（断点续跑）
  python /data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/scan_params.py --gpus 0 1 --skip_existing

  # 全部 26 类：只跑 noise、slerp、noisediffusion 的 nd_c2.8_gsqrt0.1（评测 TOPIQ/PDV）
  python /data2/home/pengfei/Graduate_experiment/NoiseDiffusion/controlnet/scan_params.py --preset quick --gpus 0 1 2
"""

import sys
import os
import argparse
import traceback
import math

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpus', nargs='+', default=['0'])
_pre_args, _ = _pre_parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpus[0]

osp = os.path
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(SCRIPT_DIR)
GRADUATE_DIR = osp.dirname(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)

from PIL import Image
import cm

# ──────────────────────────────────────────────
# 样本配置（与 run_all.py 一致）
# ──────────────────────────────────────────────
N_PROMPT = (
    'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, '
    'weird face, lopsided, disfigured, bad art, poorly drawn, low quality, '
    'drawing, blurry, faded'
)

SAMPLES = {
    'aircraft':   ('aircraft1.png',   'aircraft2.png',   'a photo of aircraft'),
    'apple':      ('apple1.png',      'apple2.png',      'a photo of apple'),
    'banana':     ('banana1.png',     'banana2.png',     'a photo of banana'),
    'bedroom':    ('bedroom1.png',    'bedroom2.png',    'a photo of bed'),
    'bee':        ('bee1.png',        'bee2.png',        'a photo,bee,wasp'),
    'bird':       ('bird1.png',       'bird2.png',       'a photo of bird'),
    'car':        ('car1.png',        'car2.png',        'a photo of car'),
    'cat':        ('cat1.png',        'cat2.png',        'a photo of cat'),
    'cherry':     ('cherry1.png',     'cherry2.png',     'a photo of cherry'),
    'cup':        ('cup1.png',        'cup2.png',        'a photo of cup'),
    'eagle':      ('eagle1.png',      'eagle2.png',      'eagle'),
    'face':       ('face1.png',       'face2.png',       'a photo of face'),
    'flower':     ('flower1.png',     'flower2.png',     'a photo of flower'),
    'forest':     ('forest1.png',     'forest2.png',     'a photo of forest'),
    'grape':      ('grape1.png',      'grape2.png',      'a photo of grape'),
    'horse':      ('horse1.png',      'horse2.png',      'a photo of a horse'),
    'house':      ('house1.png',      'house2.png',      'a photo of house'),
    'lion_tiger': ('lion_tiger1.png', 'lion_tiger2.png', "a photo of a lion's face,a photo of a tiger's face"),
    'moutain':    ('moutain1.png',    'moutain2.png',    'a photo of moutain and lake'),
    'panda':      ('panda1.png',      'panda2.png',      'a photo of panda'),
    'peach':      ('peach1.png',      'peach2.png',      'a photo of fruit,peach'),
    'pumpkin':    ('pumpkin1.png',    'pumpkin2.png',    'a photo of pumpkin'),
    'shoes':      ('shoes1.png',      'shoes2.png',      'shoes'),
    'spider':     ('spider1.png',     'spider2.png',     'a photo of spider'),
    'sushi':      ('sushi1.png',      'sushi2.png',      'a photo of sushi'),
    'tree':       ('tree1.png',       'tree2.png',       'a photo of tree'),
}

DEFAULT_CATEGORIES = ['aircraft', 'car', 'tree', 'spider', 'pumpkin', 'lion_tiger', 'cup']

# ──────────────────────────────────────────────
# 参数网格
# ──────────────────────────────────────────────
# (gamma_value, label)
GAMMA_GRID = [
    (0.0,             "g0"),
    (0.1,             "g0.1"),
    (math.sqrt(0.1),  "gsqrt0.1"),
    (math.sqrt(0.2),  "gsqrt0.2"),
]

# (coef_value | None, label, interpolation_type)
# coef=None + noisediffusion_noclip 表示不裁剪
CLIP_GRID = [
    (2.0,  "c2.0",  "noisediffusion"),
    (2.2,  "c2.2",  "noisediffusion"),
    (2.4,  "c2.4",  "noisediffusion"),
    (2.6,  "c2.6",  "noisediffusion"),
    (2.8,  "c2.8",  "noisediffusion"),
    (3.0,  "c3.0",  "noisediffusion"),
    (3.2,  "c3.2",  "noisediffusion"),
    (None, "noclip", "noisediffusion_noclip"),
]

IMG_DIR = osp.join(SCRIPT_DIR, 'sample_imgs')
IMG_SIZE = (768, 768)


def build_method_configs():
    """构建全部 (method_name, interp_type, coef, gamma) 配置列表。"""
    configs = []
    configs.append(("noise", "noise", None, None))
    configs.append(("slerp", "slerp", None, None))
    for coef_val, coef_label, interp_type in CLIP_GRID:
        for gamma_val, gamma_label in GAMMA_GRID:
            name = f"nd_{coef_label}_{gamma_label}"
            configs.append((name, interp_type, coef_val, gamma_val))
    return configs


def build_quick_configs():
    """noise + slerp + 单一 noisediffusion：c=2.8, gamma=sqrt(0.1) -> nd_c2.8_gsqrt0.1"""
    return [
        ("noise", "noise", None, None),
        ("slerp", "slerp", None, None),
        ("nd_c2.8_gsqrt0.1", "noisediffusion", 2.8, math.sqrt(0.1)),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='NoiseDiffusion 参数扫描')
    parser.add_argument(
        '--preset', choices=('full', 'quick'), default='full',
        help=(
            'full=全参数网格+noise/slerp（默认 7 类）；'
            'quick=全部类别跑 noise、slerp、noisediffusion(nd_c2.8_gsqrt0.1) 共 3 种方法'
        ),
    )
    parser.add_argument('--gpus', nargs='+', default=['0'],
                        help='GPU 编号，多个则并行（例如 --gpus 0 1 2 3）')
    parser.add_argument('--ddim_steps', type=int, default=200,
                        help='DDIM 采样步数（默认 200）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果根目录（默认依 preset 而定）')
    parser.add_argument('--guide_scale', type=float, default=10.0,
                        help='CFG 引导强度（默认 10.0）')
    parser.add_argument(
        '--categories', nargs='*', default=None,
        help='类别列表；不指定时：full 默认 7 类，quick 默认全部 26 类',
    )
    parser.add_argument('--no_eval', action='store_true',
                        help='跳过生成后的自动评测')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已有结果（断点续跑）')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        sub = 'noise_slerp_nd28' if args.preset == 'quick' else 'scan'
        args.output_dir = osp.join(SCRIPT_DIR, 'sample_results', sub)
    args.output_dir = osp.abspath(args.output_dir)
    gpus = args.gpus

    if not args.categories:
        categories = list(SAMPLES.keys()) if args.preset == 'quick' else DEFAULT_CATEGORIES
    else:
        categories = args.categories

    unknown = [c for c in categories if c not in SAMPLES]
    if unknown:
        raise ValueError(f"未知类别: {unknown}，可选: {list(SAMPLES.keys())}")

    if args.preset == 'quick':
        method_configs = build_quick_configs()
    else:
        method_configs = build_method_configs()
    nd_count = max(0, len(method_configs) - 2)

    if args.preset == 'quick':
        print(f"方法配置: {len(method_configs)} 种 (noise + slerp + nd_c2.8_gsqrt0.1)")
    else:
        print(f"方法配置: {len(method_configs)} 种 (noise + slerp + {nd_count} 种 noisediffusion)")
    print(f"类别: {categories} ({len(categories)} 个)")
    print(f"总任务数: {len(method_configs) * len(categories)}")
    print(f"preset: {args.preset}")
    print(f"GPU: {gpus}")
    print(f"输出目录: {args.output_dir}")
    if args.skip_existing:
        print("模式: 断点续跑（已有结果将被跳过）")
    print("=" * 60)

    if len(gpus) > 1:
        _dispatch_multi_gpu(args, gpus, categories)
        return

    # 单 GPU 模式
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
    print(f"正在加载模型 (GPU {gpus[0]})...")
    CM = cm.ContextManager()
    print("模型加载完成。\n")

    total = len(method_configs) * len(categories)
    done = 0
    skipped = 0
    failed = []

    for method_name, interp_type, coef_val, gamma_val in method_configs:
        for category in categories:
            done += 1
            img1_name, img2_name, prompt = SAMPLES[category]
            img1_path = osp.join(IMG_DIR, img1_name)
            img2_path = osp.join(IMG_DIR, img2_name)
            out_dir = osp.join(args.output_dir, method_name, category)

            if args.skip_existing and osp.isdir(out_dir):
                png_count = sum(1 for f in os.listdir(out_dir) if f.endswith('.png'))
                if png_count >= 7:
                    print(f"[GPU {gpus[0]}][{done}/{total}] {method_name}/{category}  [SKIP]")
                    skipped += 1
                    continue

            print(f"[GPU {gpus[0]}][{done}/{total}] {method_name}/{category}  ->  {out_dir}")

            try:
                img1 = Image.open(img1_path).resize(IMG_SIZE)
                img2 = Image.open(img2_path).resize(IMG_SIZE)
                CM.interpolate_new(
                    img1, img2,
                    prompt=prompt,
                    n_prompt=N_PROMPT,
                    ddim_steps=args.ddim_steps,
                    guide_scale=args.guide_scale,
                    out_dir=out_dir,
                    interpolation_type=interp_type,
                    custom_coef=coef_val,
                    custom_gamma=gamma_val,
                )
                print(f"  完成\n")
            except Exception as e:
                print(f"  !! 失败: {e}")
                traceback.print_exc()
                failed.append(f"{method_name}/{category}")

    print("=" * 60)
    actual_run = total - skipped
    print(f"全部完成。成功 {actual_run - len(failed)}/{actual_run}，跳过 {skipped} 个。")
    if failed:
        print(f"失败列表: {failed}")

    if not args.no_eval:
        _run_evaluation(args, categories)


def _dispatch_multi_gpu(args, gpus, categories):
    """多 GPU 并行：将类别轮流分配给各 GPU，每个 GPU 独立运行全部方法配置。"""
    import subprocess as sp

    gpu_cats = [[] for _ in gpus]
    for i, cat in enumerate(categories):
        gpu_cats[i % len(gpus)].append(cat)

    procs = []
    for gpu_id, cats in zip(gpus, gpu_cats):
        if not cats:
            continue
        cmd = [
            sys.executable, osp.abspath(__file__),
            '--preset', args.preset,
            '--gpus', gpu_id,
            '--ddim_steps', str(args.ddim_steps),
            '--output_dir', args.output_dir,
            '--guide_scale', str(args.guide_scale),
            '--categories', *cats,
            '--no_eval',
        ]
        if args.skip_existing:
            cmd.append('--skip_existing')
        print(f"[GPU {gpu_id}] 分配类别: {cats}")
        p = sp.Popen(cmd)
        procs.append(p)

    print(f"\n已启动 {len(procs)} 个并行子进程，等待全部完成...")
    print("=" * 60)
    for p in procs:
        p.wait()

    print("=" * 60)
    print("所有子进程已完成。")
    if not args.no_eval:
        _run_evaluation(args, categories)


def _run_evaluation(args, categories):
    """调用 evaluate.py 计算 TOPIQ / PDV。"""
    import subprocess
    eval_script = osp.join(GRADUATE_DIR, 'evaluate_latent', 'evaluate.py')
    if not osp.isfile(eval_script):
        print(f"\n[警告] 未找到评测脚本: {eval_script}，跳过自动评测。")
        return
    cmd = [
        sys.executable, eval_script,
        '--results_dir', args.output_dir,
        '--categories', *categories,
    ]
    print(f"\n{'='*60}")
    print("开始自动评测 TOPIQ / PDV ...")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
