"""Portable exact Python pins; discard conda build-host direct_url metadata."""
import importlib.metadata
from pathlib import Path
import sysconfig

pins = {}
for distribution in importlib.metadata.distributions(path=[sysconfig.get_paths()['purelib']]):
    name = distribution.metadata['Name']
    if name.lower().replace('_', '-') != 'agent1-dr':
        pins[name] = distribution.version
text = '# Install compiled foundation from environment.conda-explicit.txt first.\n'
text += '--extra-index-url https://download.pytorch.org/whl/cpu\n-e .\n'
text += '\n'.join(f'{name}=={pins[name]}' for name in sorted(pins, key=str.lower)) + '\n'
Path('environment.lock.txt').write_text(text)
