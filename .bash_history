
ckpt_path = "/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/mp_rank_00_model_states.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["module"] if isinstance(ckpt, dict) and "module" in ckpt else ckpt

keys = list(sd.keys())
print("num_keys:", len(keys))
print("sample keys:", keys[:30])
PY

python scripts/Evo1_server.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
clear
SRC="/hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
DST="/hpc2hdd/home/kluo573/EVO_1/InternVL3_1B_local_code"
mkdir -p "$DST"
cp -r "$SRC"/* "$DST"/
echo "✅ copied remote code to $DST"
python - <<'PY'
import pathlib, re

p = pathlib.Path("/hpc2hdd/home/kluo573/EVO_1/InternVL3_1B_local_code/modeling_intern_vit.py")
txt = p.read_text()

# 把原来的 dpr = [x.item() for x in torch.linspace(...)] 或 dpr = torch.linspace(...).tolist() 替换掉
pat = r"dpr\s*=\s*\[x\.item\(\)\s*for\s*x\s*in\s*torch\.linspace\([^\)]*\)\]\s*|dpr\s*=\s*torch\.linspace\([^\)]*\)\.tolist\(\)\s*"
rep = (
    "if config.num_hidden_layers > 1:\n"
    "            dpr = [config.drop_path_rate * i / (config.num_hidden_layers - 1) for i in range(config.num_hidden_layers)]\n"
    "        else:\n"
    "            dpr = [0.0]\n"
)

txt2, n = re.subn(pat, rep, txt, count=1)
if n == 0:
    raise SystemExit("❌ 没找到 dpr 那行，请在文件里搜索 'dpr =' 手动改。")
p.write_text(txt2)
print("✅ patched:", p)
PY

python -m py_compile model/internvl3/internvl3_embedder.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
clear
clear
nl -ba model/internvl3/internvl3_embedder.py | sed -n '65,95p'
python - <<'PY'
from pathlib import Path
import re, shutil

path = Path("model/internvl3/internvl3_embedder.py")
bak = path.with_suffix(".py.bak")
shutil.copy2(path, bak)

txt = path.read_text()

# 1) 在 self.transform 后插入 local_model_path + 覆盖 model_name
needle = "self.transform = build_transform(image_size)"
insert = (
    "self.transform = build_transform(image_size)\n"
    "        local_model_path = \"/hpc2hdd/home/kluo573/EVO_1/InternVL3_1B_local_code\"\n"
    "        model_name = local_model_path  # force use local code\n"
)
if needle in txt and "force use local code" not in txt:
    txt = txt.replace(needle, insert)

# 2) 确保 use_flash_attn=False 后面有逗号（你贴的版本少逗号会炸）
txt = txt.replace("use_flash_attn=False\n", "use_flash_attn=False,\n")

path.write_text(txt)
print("✅ patched", path, "backup ->", bak)
PY

python -m py_compile model/internvl3/internvl3_embedder.py
python - <<'PY'
from pathlib import Path
import shutil, re

p = Path("/hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/modeling_intern_vit.py")
bak = p.with_suffix(".py.bak")
shutil.copy2(p, bak)

txt = p.read_text()

# 把 dpr 的 torch.linspace + item 彻底换成纯 Python（100% 不会 meta）
pattern = r"dpr\s*=\s*\[x\.item\(\)\s*for\s*x\s*in\s*torch\.linspace\(\s*0\s*,\s*config\.drop_path_rate\s*,\s*config\.num_hidden_layers\s*\)\s*\]"
replacement = (
    "if config.num_hidden_layers > 1:\n"
    "            dpr = [config.drop_path_rate * i / (config.num_hidden_layers - 1) for i in range(config.num_hidden_layers)]\n"
    "        else:\n"
    "            dpr = [0.0]"
)

new_txt, n = re.subn(pattern, replacement, txt, count=1)
if n == 0:
    raise SystemExit("❌ 没找到目标 dpr 行（可能格式不同）。请在文件里搜索 'dpr =' 把那行贴我。")

p.write_text(new_txt)
print("✅ patched", p, "backup ->", bak)
PY

python scripts/Evo1_server.py
pip install -U sentencepiece tiktoken
local_model_path = "/hpc2hdd/home/kluo573/EVO_1/InternVL3_1B_local_code"
model_name = local_model_path  # force use local code
python - <<'PY'
from pathlib import Path
import re, shutil

path = Path("model/internvl3/internvl3_embedder.py")
bak = path.with_suffix(".py.fix2.bak")
shutil.copy2(path, bak)

txt = path.read_text()
txt = re.sub(r"\n\s*local_model_path\s*=\s*\"/hpc2hdd/home/kluo573/EVO_1/InternVL3_1B_local_code\"\n\s*model_name\s*=\s*local_model_path\s*# force use local code\n", "\n", txt)
path.write_text(txt)
print("✅ removed forced local_model_path; backup ->", bak)
PY

nl -ba model/internvl3/internvl3_embedder.py | sed -n '72,90p'
python scripts/Evo1_server.py
conda activate Evo1
python -c "import transformers; print(transformers.__version__)"
# 如果这里显示 5.x，那就按下面做
pip uninstall -y transformers
pip install -U "transformers==4.46.3" "tokenizers==0.20.3" "accelerate==0.34.2" "safetensors>=0.4.5"
rm -rf ~/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B
python scripts/Evo1_server.py
 conda install -n libero -c conda-forge cmake -y
 python3 -c "
import torch
ckpt = torch.load('/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/mp_rank_00_model_states.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
if 'module' in ckpt:
    print('\\nModule keys:', list(ckpt['module'].keys())[:20], '...')
    print('Total params:', len(ckpt['module']))
"
 ls -lh /hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/
 conda run -n Evo1 python -c "
import torch
ckpt = torch.load('/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/mp_rank_00_model_states.pt', map_location='cpu')
print('Top-level keys:', list(ckpt.keys()))
if 'module' in ckpt:
    print('\nModule keys (first 30):', list(ckpt['module'].keys())[:30])
    print('\nTotal module parameters:', len(ckpt['module']))
"
 conda run -n Evo1 python -c "
import torch
ckpt = torch.load('/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
keys = list(ckpt['module'].keys())

# 统计不同模块
vit_keys = [k for k in keys if 'vision' in k]
vlm_keys = [k for k in keys if 'language' in k.lower() or 'llm' in k.lower() or 'qwen' in k.lower()]
action_keys = [k for k in keys if 'action' in k or 'head' in k or 'flow' in k]

print('=== ViT 参数 (前10个) ===')
print('\n'.join(vit_keys[:10]))
print(f'\n总计 ViT 参数: {len(vit_keys)}')

print('\n=== VLM/语言模型参数 (前10个) ===')
print('\n'.join(vlm_keys[:10]) if vlm_keys else '未找到')
print(f'总计 VLM 参数: {len(vlm_keys)}')

print('\n=== Action 层参数 (所有) ===')
print('\n'.join(action_keys) if action_keys else '未找到')
print(f'总计 Action 参数: {len(action_keys)}')

print(f'\n总参数数: {len(keys)}')
"
 conda run -n Evo1 python -c "
import torch
ckpt = torch.load('/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO/mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
keys = list(ckpt['module'].keys())

# 统计 vision encoder 层数
vision_layers = set()
for k in keys:
    if 'vision_model.encoder.layers.' in k:
        layer_num = k.split('vision_model.encoder.layers.')[1].split('.')[0]
        if layer_num.isdigit():
            vision_layers.add(int(layer_num))

# 统计 language model 层数
language_layers = set()
for k in keys:
    if 'language_model.model.layers.' in k:
        layer_num = k.split('language_model.model.layers.')[1].split('.')[0]
        if layer_num.isdigit():
            language_layers.add(int(layer_num))

print(f'Vision Encoder (ViT) 层数: {len(vision_layers)} (层编号: {sorted(vision_layers)})')
print(f'Language Model 层数: {len(language_layers)} (层编号: {sorted(language_layers)})')
print(f'Action Head Transformer 层数: 8 (从之前输出可见 0-7)')
"
 sed -n '308,318p' /hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/modeling_intern_vit.py
 sed -i 's/dpr = \[x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)\]/dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers).tolist()/' /hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/modeling_intern_vit.py
 sed -n '312p' /hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/modeling_intern_vit.py
 sed -i "s/dpr = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers).tolist()/dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]/" /hpc2hdd/home/kluo573/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVL3_hyphen_1B/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/modeling_intern_vit.py
 mkdir -p /hpc2hdd/home/kluo573/tmp /hpc2hdd/home/kluo573/.cache/pip && TMPDIR=/hpc2hdd/home/kluo573/tmp PIP_CACHE_DIR=/hpc2hdd/home/kluo573/.cache/pip MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
 mkdir -p /hpc2hdd/home/kluo573/tmp /hpc2hdd/home/kluo573/.cache/pip && TMPDIR=/hpc2hdd/home/kluo573/tmp PIP_CACHE_DIR=/hpc2hdd/home/kluo573/.cache/pip MAX_JOBS=64 conda run -n Evo1 pip install -v flash-attn --no-build-isolation
 conda run -n Evo1 python - <<'PY'
import importlib.util
spec = importlib.util.find_spec('flash_attn')
print('flash_attn_installed', bool(spec))
PY

 echo ok
git clone https://github.com/MINT-SJTU/Evo-1.git
cd Evo-1/
conda create -n Evo1 python=3.10 -y
conda activate Evo1
cd Evo_1
pip install -r requirements.txt
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
mkdir -p /hpc2hdd/home/kluo573/tmp /hpc2hdd/home/kluo573/.cache/pip && TMPDIR=/hpc2hdd/home/kluo573/tmp PIP_CACHE_DIR=/hpc2hdd/home/kluo573/.cache/pip MAX_JOBS=64 conda run -n Evo1 pip install -v flash-attn --no-build-isolation
conda create -n libero python=3.8.13 -y
conda create -n libero python=3.8.13 -y
conda deactivate
conda create -n libero python=3.8.13 -y
conda activate libero
cd LIBERO_evaluation/
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
conda install -n libero -c conda-forge cmake -y
conda run -n libero cmake --version
pip install -r requirements.txt
conda install -n libero -c conda-forge 'cmake>=3.18' -y
pip install -r requirements.txt
cmake --version
conda install -n libero -c conda-forge cmake=3.27 -y
conda run -n libero bash -c "CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install egl-probe"
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
pip install websockets
pip install huggingface_hub
hf download MINT-SJTU/Evo1_LIBERO --local-dir /path/to/save/checkpoint/
hf download MINT-SJTU/Evo1_LIBERO --local-dir /path/to/save/checkpoint/
hf download MINT-SJTU/Evo1_LIBERO --local-dir ~/EVO_1/Evo-1/LIBERO_evaluation/checkpoints/Evo1_LIBERO
cd LIBERO_evaluation
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation
python libero_client_4tasks.py
clear
python scripts/Evo1_server.py
clear
python libero_client_4tasks.py
conda install -c conda-forge mesalib osmesa pyopengl -y
python libero_client_4tasks.py
python libero_client_4tasks.py
clear
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
python libero_client_4tasks.py
python libero_client_4tasks.py
conda search -c conda-forge "mesa-libosmesa"
conda search -c conda-forge "libosmesa"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe python libero_client_4tasks.py
python - <<'PY'
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
PY

python - <<'PY'
import ctypes
for lib in ["libOSMesa.so.8", "libOSMesa.so", "libGL.so.1"]:
    try:
        ctypes.CDLL(lib)
        print("OK:", lib)
    except OSError as e:
        print("MISSING:", lib, "->", e)
PY

conda install -c conda-forge libosmesa
sudo apt-get update
sudo apt-get install libosmesa6
ls /usr/lib/x86_64-linux-gnu/libosmesa*
sudo find / -name "libosmesa*"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
python libero_client_4tasks.py
pip install nvitop
nvitop
nvitop
conda activate libero
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation
python mt50_evo1_client_prompt.py
python libero_client_4tasks.py
python libero_client_4tasks.py
conda activate Evo1
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/Evo_1
python scripts/Evo1_server.py
python scripts/Evo1_server.py
lsof -i :9000
netstat -tulpn | grep :9000
kill -9 521825
netstat -tulpn | grep :9000
python scripts/Evo1_server.py
python scripts/Evo1_server.py
stty cols 112
 stty rows 25
  export TERM=xterm
 clear

stty cols 112
 stty rows 25
  export TERM=xterm
 clear
ssh-keygen -t rsa -b 4096
ssh-keygen -t rsa -b 4096
ssh-copy-id -i ~/.ssh/id_rsa_new.pub kluo573-A40One@10.120.18.240 -p 6988
ssh-copy-id -i ~/.ssh/id_rsa_new.pub -p 6988 kluo573-A40One@10.120.18.240
conda envs
conda env
envs
env
conda env list
conda activate qwen_env
cd /hpc2hdd/home/kluo573/qwen3_0.6b_learning
conda activate qwen_env
conda activate /hpc2hdd/home/kluo573/qwen3_0.6b_learning/qwen_env
pythonloop_qwen_0.6B.py
python loop_qwen-0.6B.py
python loop_qwen_0.6B.py
python run_monitor.py
nvitop
nvitop
conda activate /hpc2hdd/home/kluo573/qwen3_0.6b_learning/qwen_env
python flux_testpy
python flux_test.py
cd /hpc2hdd/home/kluo573/qwen3_0.6b_learning
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python flux_test.py
python /hpc2hdd/home/kluo573/qwen3_0.6b_learning/run_monitor.py
nvitop
conda env list
conda activate /hpc2hdd/home/kluo573/qwen3_0.6b_learning/qwen_env
pyhton run_monitor.py
python run_monitor.py
cd /hpc2hdd/home/kluo573/qwen3_0.6b_learning
python run_monitor.py
conda env list
conda activate /hpc2hdd/home/kluo573/qwen3_0.6b_learning/qwen_env
cd   /hpc2hdd/home/kluo573/qwen3_0.6b_learning
python  run_monitor.py
conda activate smoothquant
conda create -n smoothquant python=3.8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip uninstall torch torchvision torchaudio -y
python -c "import torch" 2>&1
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.36.0 accelerate datasets zstandard
python setup.py install
git clone https://github.com/Kevin-the-Great/smoothquant.git
cd smoothquant
python setup.py install
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from smoothquant.smooth import smooth_lm; print('SmoothQuant 导入成功')"
python -c "
import torch
print(f'GPU 数量: {torch.cuda.device_count()}')
print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
print(f'显存大小: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
python -c "print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
python -c "
import torch
python -c "print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
python -c "
import torch
print(f'GPU 数量: {torch.cuda.device_count()}')
print(f'GPU 名称: {torch.cuda.get_device_name(0)}')python -c "print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
python -c "
import torch
print(f'GPU 数量: {torch.cuda.device_count()}')
print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
print(f'显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
conda activate smoothquant
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
conda activate smoothquant
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
/hpc2hdd/home/kluo573/.conda/envs/smoothquant/bin/python /hpc2hdd/home/kluo573/smoothquant/test.py
nvitop
conda activate smoothquant
conda deactivate
conda activate libero
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation
python libero_client_4tasks.py
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python libero_client_4tasks.py
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python libero_client_4tasks.py
(libero) kluo573@cb0a15a213e1:~/EVO_1/Evo-1/LIBERO_evaluation$ export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
python libero_client_4tasks.py
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Traceback (most recent call last):
  File "libero_client_4tasks.py", line 19, in <module>
    from libero.libero.envs import OffScreenRenderEnv
  File "/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/LIBERO/libero/libero/envs/__init__.py", line 1, in <module>
    from .bddl_base_domain import TASK_MAPPING
  File "/hpc2hdd/home/kluo573/EVO_1/Evo-1/LIBERO_evaluation/LIBERO/libero/libero/envs/bddl_base_domain.py", line 3, in <module>
    import robosuite.utils.transform_utils as T
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/robosuite/__init__.py", line 11, in <module>
    from robosuite.environments.base import make
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/robosuite/environments/__init__.py", line 1, in <module>
    from .base import REGISTERED_ENVS, MujocoEnv
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/robosuite/environments/base.py", line 12, in <module>
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/robosuite/utils/binding_utils.py", line 12, in <module>
    import mujoco
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/mujoco/__init__.py", line 48, in <module>
    from mujoco.gl_context import *
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/mujoco/gl_context.py", line 38, in <module>
    from mujoco.osmesa import GLContext as _GLContext
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/mujoco/osmesa/__init__.py", line 31, in <module>
    from OpenGL import GL
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/OpenGL/GL/__init__.py", line 4, in <module>
    from OpenGL.GL.VERSION.GL_1_1 import *
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/OpenGL/GL/VERSION/GL_1_1.py", line 14, in <module>
    from OpenGL.raw.GL.VERSION.GL_1_1 import *
  File "/hpc2hdd/home/kluo573/.conda/envs/libero/lib/python3.8/site-packages/OpenGL/raw/GL/VERSION/GL_1_1.py", line 7, in <module>
    from OpenGL.r
clear
conda activate smoothquant
conda deactivate
conda env list
conda activate Evo1
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1
python scripts/Evo1_server.py
cd /hpc2hdd/home/kluo573/EVO_1
python scripts/Evo1_server.py
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/Evo_1
python scripts/Evo1_server.py
echo "Script written"
conda activate libero
cd ~/EVO_1/Evo-1/LIBERO_evaluation
python libero_client_4tasks.py
python libero_client_4tasks.py
conda activate Evo1
cd /hpc2hdd/home/kluo573/EVO_1/Evo-1/Evo_1
python scripts/Evo1_server.py
python scripts/Evo1_server.py
python scripts/Evo1_server.py
nohup python scripts/Evo1_server.py > server.log 2>&1 &
(Evo1) kluo573@cb0a15a213e1:~/EVO_1/Evo-1/Evo_1$ nohup python scripts/Evo1_server.py > server.log 2>&1 &
[1] 840427
nohup python scripts/Evo1_server.py > server.log 2>&1 &
clear
tail -f ~/EVO_1/Evo-1/Evo_1/server.log
