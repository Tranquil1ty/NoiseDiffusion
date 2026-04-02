"""
批量插值启动脚本
用法示例：
  # 用 noisediffusion 方法跑所有类别
  python controlnet/run_all.py --interpolation noisediffusion

  # 用 noise 和 slerp 方法，自定义 ddim_steps 和输出目录
  python controlnet/run_all.py --interpolation noise slerp --ddim_steps 150 --output_dir results/exp1

  # 跑全部三种方法
  python controlnet/run_all.py --interpolation all

  # 只跑指定类别
  python controlnet/run_all.py --interpolation noisediffusion --categories apple cat bird

输出目录结构：
  {output_dir}/{interpolation_type}/{category}/001.png ... 009.png
"""

import sys
import os
import argparse
import traceback

# 必须在 import torch/cm 之前设置，否则 CUDA 已初始化后设置无效
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _pre_parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

osp = os.path
sys.path.append(osp.dirname(osp.abspath(__file__)))

from PIL import Image
import cm

# ──────────────────────────────────────────────
# 所有样本配置：类别名 -> (img1路径, img2路径, prompt)
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

ALL_METHODS = ['noise', 'slerp', 'noisediffusion', 'noisediffusion_noclip', 'noisediffusion_gamma0']

IMG_DIR = 'controlnet/sample_imgs'
IMG_SIZE = (768, 768)


def parse_args():
    parser = argparse.ArgumentParser(description='批量图像插值生成脚本')
    parser.add_argument(
        '--interpolation', nargs='+',
        default=['noisediffusion'],
        help=(
            '插值方式，可选：noise / slerp / noisediffusion / all。'
            '支持同时指定多个，例如 --interpolation noise slerp'
        ),
    )
    parser.add_argument(
        '--ddim_steps', type=int, default=200,
        help='DDIM 采样步数（默认 200）'
    )
    parser.add_argument(
        '--output_dir', type=str, default='controlnet/sample_results',
        help='结果根目录（默认 controlnet/sample_results）'
    )
    parser.add_argument(
        '--guide_scale', type=float, default=10.0,
        help='CFG 引导强度（默认 10.0）'
    )
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='使用的 GPU 编号（默认 0）'
    )
    parser.add_argument(
        '--categories', nargs='+',
        default=['aircraft', 'car', 'tree', 'spider', 'pumpkin', 'lion_tiger', 'cup'],
        help='运行的类别列表（默认 7 个：aircraft car tree spider pumpkin lion_tiger cup）'
    )
    return parser.parse_args()


def resolve_methods(interpolation_args):
    methods = []
    for m in interpolation_args:
        if m == 'all':
            methods.extend(ALL_METHODS)
        elif m in ALL_METHODS:
            methods.append(m)
        else:
            raise ValueError(
                f"未知插值方式 '{m}'，可选：{ALL_METHODS} 或 all"
            )
    seen = set()
    return [x for x in methods if not (x in seen or seen.add(x))]


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    methods = resolve_methods(args.interpolation)
    categories = args.categories if args.categories is not None else list(SAMPLES.keys())

    # 验证类别是否存在
    unknown = [c for c in categories if c not in SAMPLES]
    if unknown:
        raise ValueError(f"未知类别：{unknown}，可选：{list(SAMPLES.keys())}")

    print(f"插值方式: {methods}")
    print(f"类别数量: {len(categories)}")
    print(f"ddim_steps: {args.ddim_steps}")
    print(f"输出目录: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print("=" * 60)

    # 加载模型（只加载一次）
    print("正在加载模型...")
    CM = cm.ContextManager()
    print("模型加载完成。\n")

    total = len(methods) * len(categories)
    done = 0
    failed = []

    for method in methods:
        for category in categories:
            done += 1
            img1_name, img2_name, prompt = SAMPLES[category]
            img1_path = osp.join(IMG_DIR, img1_name)
            img2_path = osp.join(IMG_DIR, img2_name)

            # 输出到 {output_dir}/{method}/{category}/
            out_dir = osp.join(args.output_dir, method, category)

            print(f"[{done}/{total}] {method}/{category}  ->  {out_dir}")

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
                    interpolation_type=method,
                )
                print(f"  完成\n")
            except Exception as e:
                print(f"  !! 失败：{e}")
                traceback.print_exc()
                failed.append(f"{method}/{category}")

    print("=" * 60)
    print(f"全部完成。成功 {total - len(failed)}/{total} 个任务。")
    if failed:
        print(f"失败列表：{failed}")

    _run_evaluation(args, methods, categories)


def _run_evaluation(args, methods, categories):
    """生成完成后自动调用 evaluate.py 计算 TOPIQ / PDV。"""
    import subprocess
    controlnet_dir = osp.dirname(osp.abspath(__file__))
    graduate_dir = osp.dirname(osp.dirname(controlnet_dir))
    eval_script = osp.join(graduate_dir, 'evaluate_latent', 'evaluate.py')
    if not osp.isfile(eval_script):
        print(f"\n[警告] 未找到评测脚本：{eval_script}，跳过自动评测。")
        return
    cmd = [
        sys.executable, eval_script,
        '--results_dir', osp.abspath(args.output_dir),
        '--methods', *methods,
        '--categories', *categories,
    ]
    print(f"\n{'='*60}")
    print("开始自动评测 TOPIQ / PDV ...")
    print(f"{'='*60}")
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
