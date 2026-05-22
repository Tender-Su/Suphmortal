from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes


AFFINITY_ENV_VAR = 'MORTAL_CPU_AFFINITY'
DEFAULT_AFFINITY_SPEC = 'disabled'
DISABLED_AFFINITY_VALUES = frozenset({'', '0', 'false', 'off', 'none', 'disable', 'disabled'})
P_CORE_AFFINITY_VALUES = frozenset({
    'p_cores',
    'p-cores',
    'pcores',
    'performance',
    'performance_cores',
    'performance-cores',
    'big',
    'big_cores',
    'big-cores',
})
ALL_CORE_AFFINITY_VALUES = frozenset({'all', 'any'})

ERROR_INSUFFICIENT_BUFFER = 122
RELATION_PROCESSOR_CORE = 0


if os.name == 'nt':
    DWORD_PTR = ctypes.c_size_t
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(DWORD_PTR),
        ctypes.POINTER(DWORD_PTR),
    )
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, DWORD_PTR)
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.GetLogicalProcessorInformationEx.argtypes = (
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL


    class SystemLogicalProcessorInformationExHeader(ctypes.Structure):
        _fields_ = [
            ('Relationship', wintypes.DWORD),
            ('Size', wintypes.DWORD),
        ]


    class ProcessorRelationshipHeader(ctypes.Structure):
        _fields_ = [
            ('Flags', ctypes.c_ubyte),
            ('EfficiencyClass', ctypes.c_ubyte),
            ('Reserved', ctypes.c_ubyte * 20),
            ('GroupCount', wintypes.WORD),
        ]


    class GroupAffinity(ctypes.Structure):
        _fields_ = [
            ('Mask', DWORD_PTR),
            ('Group', wintypes.WORD),
            ('Reserved', wintypes.WORD * 3),
        ]


def is_affinity_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in DISABLED_AFFINITY_VALUES


def ensure_affinity_env_default(default: str | None = None) -> str:
    if AFFINITY_ENV_VAR not in os.environ:
        if default is not None:
            os.environ[AFFINITY_ENV_VAR] = default
            return default
        return ''
    return os.environ.get(AFFINITY_ENV_VAR, '')


def mask_to_logical_cpus(mask: int) -> list[int]:
    return [bit for bit in range(mask.bit_length()) if mask & (1 << bit)]


def parse_cpu_list_spec(spec: str) -> int:
    mask = 0
    for part in spec.split(','):
        token = part.strip()
        if not token:
            continue
        if '-' in token:
            start_text, end_text = token.split('-', maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f'invalid CPU range: {token}')
            for logical_cpu in range(start, end + 1):
                mask |= 1 << logical_cpu
            continue
        mask |= 1 << int(token)
    if mask <= 0:
        raise ValueError(f'invalid CPU affinity spec: {spec!r}')
    return mask


def resolve_affinity_mask(
    spec: str,
    *,
    allowed_mask: int,
    efficiency_class_masks: dict[int, int] | None = None,
) -> tuple[int, str]:
    normalized = spec.strip().lower()
    if normalized in ALL_CORE_AFFINITY_VALUES:
        return allowed_mask, 'all logical CPUs'

    if normalized in P_CORE_AFFINITY_VALUES:
        efficiency_class_masks = dict(efficiency_class_masks or {})
        if not efficiency_class_masks:
            return allowed_mask, 'missing Windows efficiency-class data; kept current process mask'
        if len(efficiency_class_masks) == 1:
            return allowed_mask, 'single efficiency class detected; kept current process mask'
        p_core_class = max(efficiency_class_masks)
        return allowed_mask & efficiency_class_masks[p_core_class], f'Windows efficiency class {p_core_class}'

    if normalized.startswith('0x'):
        return allowed_mask & int(normalized, 16), f'explicit affinity mask {normalized}'

    return allowed_mask & parse_cpu_list_spec(normalized), f'explicit logical CPU set {normalized}'


def get_process_affinity_mask() -> int:
    if os.name != 'nt':
        return 0
    process_mask = DWORD_PTR()
    system_mask = DWORD_PTR()
    current_process = kernel32.GetCurrentProcess()
    if not kernel32.GetProcessAffinityMask(
        current_process,
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(process_mask.value)


def set_process_affinity_mask(mask: int) -> None:
    current_process = kernel32.GetCurrentProcess()
    if not kernel32.SetProcessAffinityMask(current_process, DWORD_PTR(mask)):
        raise ctypes.WinError(ctypes.get_last_error())


def detect_efficiency_class_masks() -> dict[int, int]:
    if os.name != 'nt':
        return {}

    required = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE,
        None,
        ctypes.byref(required),
    )
    last_error = ctypes.get_last_error()
    if last_error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(last_error)

    raw = (ctypes.c_byte * required.value)()
    if not kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE,
        ctypes.byref(raw),
        ctypes.byref(required),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    header_size = ctypes.sizeof(SystemLogicalProcessorInformationExHeader)
    processor_header_size = ctypes.sizeof(ProcessorRelationshipHeader)
    group_affinity_size = ctypes.sizeof(GroupAffinity)

    masks_by_efficiency_class: dict[int, int] = {}
    offset = 0
    while offset < required.value:
        info = SystemLogicalProcessorInformationExHeader.from_buffer_copy(raw, offset)
        if info.Relationship == RELATION_PROCESSOR_CORE:
            processor_info = ProcessorRelationshipHeader.from_buffer_copy(raw, offset + header_size)
            group_offset = offset + header_size + processor_header_size
            for group_index in range(int(processor_info.GroupCount)):
                group_affinity = GroupAffinity.from_buffer_copy(
                    raw,
                    group_offset + group_index * group_affinity_size,
                )
                if int(group_affinity.Group) != 0:
                    continue
                group_mask = int(group_affinity.Mask)
                if group_mask <= 0:
                    continue
                efficiency_class = int(processor_info.EfficiencyClass)
                masks_by_efficiency_class[efficiency_class] = (
                    masks_by_efficiency_class.get(efficiency_class, 0) | group_mask
                )
        offset += int(info.Size)
    return masks_by_efficiency_class


def maybe_configure_process_affinity(*, log: bool = True, context: str = 'python process') -> dict[str, object]:
    requested = os.environ.get(AFFINITY_ENV_VAR, '')
    result: dict[str, object] = {
        'requested': requested,
        'applied': False,
        'changed': False,
        'reason': 'disabled',
        'mask': None,
        'logical_cpus': [],
    }
    if not is_affinity_enabled(requested):
        return result
    if os.name != 'nt':
        result['reason'] = 'CPU affinity helper is only implemented on Windows'
        if log:
            logging.warning('%s requested %s=%s but this platform is not Windows.', context, AFFINITY_ENV_VAR, requested)
        return result

    try:
        current_mask = get_process_affinity_mask()
        if current_mask <= 0:
            raise RuntimeError('current process affinity mask is empty')
        efficiency_class_masks = None
        if requested.strip().lower() in P_CORE_AFFINITY_VALUES:
            efficiency_class_masks = detect_efficiency_class_masks()
        target_mask, reason = resolve_affinity_mask(
            requested,
            allowed_mask=current_mask,
            efficiency_class_masks=efficiency_class_masks,
        )
        if target_mask <= 0:
            raise RuntimeError(f'affinity spec {requested!r} resolved to an empty mask')
        changed = target_mask != current_mask
        if changed:
            set_process_affinity_mask(target_mask)
        logical_cpus = mask_to_logical_cpus(target_mask)
        result.update({
            'applied': True,
            'changed': changed,
            'reason': reason,
            'mask': target_mask,
            'logical_cpus': logical_cpus,
        })
        if log:
            action = 'Pinned' if changed else 'Kept'
            logging.info(
                '%s %s to logical CPUs %s via %s=%s (%s, mask=%s).',
                action,
                context,
                logical_cpus,
                AFFINITY_ENV_VAR,
                requested,
                reason,
                hex(target_mask),
            )
        return result
    except Exception as exc:
        result['reason'] = str(exc)
        if log:
            logging.warning(
                'Failed to apply %s=%s for %s: %s',
                AFFINITY_ENV_VAR,
                requested,
                context,
                exc,
            )
        return result
