import pathlib


def _default_demo_names():
    return [
        'gemma3',
        'LiquidAI-LFM2.5-230M',
        'moonshine',
        'moonshine_streaming',
        'LiquidAI-LFM2-VL-450M',
        'object_detection',
        'pose_estimation',
    ]


def _read_demo_config():
    config = {}
    candidates = [
        pathlib.Path(__file__).resolve().with_name('config.ini'),
        pathlib.Path.cwd() / 'config.ini',
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
        if config:
            return config
    return config


def _effective_demo_names():
    config = _read_demo_config()
    chosen = []
    for name in _default_demo_names():
        state = config.get(name, 'Y').strip().upper()
        if state == 'Y':
            chosen.append(name)
    return chosen or _default_demo_names()


print(','.join(_effective_demo_names()))
