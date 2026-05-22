import mortal.core.prelude

import os
import sys
import json
import torch
from datetime import datetime, timezone
from mortal.core.model import Brain, CategoricalPolicy, GRP
from mortal.eval.engine import MortalEngine
from mortal.core.common import filtered_trimmed_lines
from libriichi.mjai import Bot
from libriichi.dataset import Grp
from mortal.config import config
from mortal.core.checkpoint_utils import checkpoint_brain_is_oracle_structure, load_brain_state_with_input_bridge
from mortal.eval.search_runtime import build_search_runtime_bundle_from_state

USAGE = '''Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3].'''

def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'

    device = torch.device('cpu')
    state = torch.load(config['control']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']
    if 'tag' in state:
        tag = state['tag']
    else:
        time = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{time}'

    mortal = Brain(
        version=version,
        num_blocks=num_blocks,
        conv_channels=conv_channels,
        is_oracle=checkpoint_brain_is_oracle_structure(state),
        Norm="GN",
    ).eval()
    dqn = CategoricalPolicy().eval()
    load_brain_state_with_input_bridge(mortal, state['mortal'])
    dqn.load_state_dict(state.get('policy_net') or state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = not review_mode,
        enable_rule_based_agari_guard = True,
        name = 'mortal',
        search_runtime_bundle = build_search_runtime_bundle_from_state(
            state,
            device=device,
            enable_compile=False,
        ),
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []
    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if reaction := bot.react(line):
            print(reaction, flush=True)
        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    if review_mode:
        grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        grp.load_state_dict(grp_state['model'])
        grp_dtype = next(grp.parameters()).dtype

        ins = Grp.load_log('\n'.join(logs))
        feature = ins.take_feature()
        seq = list(map(
            lambda idx: torch.as_tensor(feature[:idx+1], dtype=grp_dtype, device=device),
            range(len(feature)),
        ))

        with torch.inference_mode():
            logits = grp(seq)
        matrix = grp.calc_matrix(logits)
        extra_data = {
            'model_tag': tag,
            'phi_matrix': matrix.tolist(),
        }
        print(json.dumps(extra_data), flush=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
