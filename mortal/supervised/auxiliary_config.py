"""Auxiliary configuration shared by SL bootstrap and checkpoint restoration."""
from copy import deepcopy


def resolve_effective_config_section(config, config_section):
    if not isinstance(config, dict):
        return {}
    section_cfg = config.get(config_section)
    return section_cfg if isinstance(section_cfg, dict) else {}


def resolve_effective_aux_cfg(config, config_section):
    if not isinstance(config, dict):
        return {}
    base = config.get('aux', {})
    effective = dict(base) if isinstance(base, dict) else {}
    scoped = resolve_effective_config_section(config, config_section).get('aux', {})
    if isinstance(scoped, dict):
        # Match training's section-level shallow override, including nested tables.
        effective.update(scoped)
    return effective


def auxiliary_recipe_from_config(config, *, config_section='supervised'):
    if not isinstance(config, dict):
        raise ValueError('auxiliary config must be a mapping')
    base = config.get('aux', {})
    section = config.get(config_section, {})
    if not isinstance(base, dict) or not isinstance(section, dict):
        raise ValueError('auxiliary config sections must be mappings')
    scoped = section.get('aux', {})
    rank = section.get('rank_aux', {})
    if not isinstance(scoped, dict) or not isinstance(rank, dict):
        raise ValueError(f'{config_section}.aux and rank_aux must be mappings')
    return deepcopy({'aux': base, 'section_aux': scoped, 'rank_aux': rank})


def effective_auxiliary_recipe(config, *, config_section='supervised'):
    recipe = auxiliary_recipe_from_config(config, config_section=config_section)
    return {
        'aux': {**recipe['aux'], **recipe['section_aux']},
        'rank_aux': recipe['rank_aux'],
    }


def auxiliary_step_offset_from_state(state, *, optimizer_steps, source_optimizer_steps):
    """Preserve the auxiliary ramp clock even when the main optimizer is reset."""
    saved_steps = state.get('auxiliary_optimizer_steps', source_optimizer_steps)
    if not isinstance(saved_steps, int) or isinstance(saved_steps, bool) or saved_steps < 0:
        raise ValueError('saved auxiliary optimizer steps must be a non-negative integer')
    if optimizer_steps < 0:
        raise ValueError('optimizer steps must be non-negative')
    return saved_steps - optimizer_steps


def validate_exact_resume_auxiliary_recipe(state, config, *, config_section='supervised'):
    """An exact resume cannot silently change loss coefficients or target settings.

    We compare saved settings after resolving section overrides. Intentionally
    different objectives belong in a new weights-only branch. Legacy checkpoints
    without a recipe cannot establish the exact-resume contract.
    """
    saved_config = state.get('config')
    if not isinstance(saved_config, dict):
        raise RuntimeError('exact SL resume requires a saved auxiliary config')
    saved_section = state.get('config_section') or config_section
    if saved_section != config_section:
        raise RuntimeError('exact SL resume requires the same config section')
    saved = effective_auxiliary_recipe(saved_config, config_section=config_section)
    current = effective_auxiliary_recipe(config, config_section=config_section)
    changed = [key for key in saved if saved[key] != current[key]]
    if changed:
        raise RuntimeError(
            f'{config_section}.state_file auxiliary recipe mismatch: '
            + ', '.join(changed)
            + f'. Use {config_section}.init_state_file in a new run for an '
            'intentional objective change; do not rewrite the saved checkpoint.'
        )
