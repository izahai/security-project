"""Pre-flight check for the AEGIS training entry point.

Reads AEGIS.py as source rather than importing it, because the point is to run
on the box where you edit the config -- which has no GPU and no torch. Once the
VM env exists, `python train-scripts/AEGIS.py --help` supersedes this.

    python setup/check_config.py
"""
import argparse
import ast
import os
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / 'train-scripts' / 'AEGIS.py'
tree = ast.parse(SRC.read_text(encoding='utf-8'))

# 1. W&B key comes from the environment, never a literal (AGENTS.md), and its
#    absence disables W&B instead of raising at import time.
guard = next(n for n in tree.body if isinstance(n, ast.If) and 'WANDB_API_KEY' in ast.unparse(n))
os.environ.pop('WANDB_API_KEY', None)
os.environ.pop('WANDB_MODE', None)
exec(compile(ast.Module([guard], []), '<guard>', 'exec'),
     {'os': os, 'wandb': type('stub', (), {'login': staticmethod(lambda **kw: None)})})
assert os.environ.get('WANDB_MODE') == 'disabled', 'no key should disable W&B, not crash'

# 2. Rebuild the CLI and exercise the flags that decide the run.
parser = argparse.ArgumentParser()
for call in (n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and ast.unparse(n.func).endswith('add_argument')):
    kw = {k.arg: eval(ast.unparse(k.value)) for k in call.keywords if k.arg != 'help'}
    kw.pop('required', None)
    parser.add_argument(ast.literal_eval(call.args[0]), **kw)

args = parser.parse_args(['--prompt', 'church', '--lr', '1e-5', '--save_interval', '1000'])
assert args.lr == 1e-5, f'--lr must accept a float, got {args.lr!r}'
assert args.save_interval == 1000, 'default 200 writes 5 checkpoints/concept (~115 GB for three)'
assert (args.gp_mu, args.gp_w_cap) == (0.1, 1.0), \
    f'GRP defaults should follow the paper (mu=0.1, omega cap 1), got {args.gp_mu}, {args.gp_w_cap}'

released = parser.parse_args(['--prompt', 'church', '--gp_mu', '0.2', '--gp_w_cap', '1e6'])
assert (released.gp_mu, released.gp_w_cap) == (0.2, 1e6), 'released-code variant must stay reachable'

print('ok  W&B key from env; absent key disables W&B')
print('ok  --lr 1e-5 parses as float')
print('ok  GRP defaults mu=0.1 w_cap=1.0; --gp_mu 0.2 --gp_w_cap 1e6 reproduces the released code')
