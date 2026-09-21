"""Check the staged/public file set without printing sensitive matched values."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = set(subprocess.check_output(['git', 'show', ':PUBLIC_FILES.txt'], cwd=ROOT).decode().splitlines())
tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')) - {''}
errors = []
for name in sorted(tracked):
    if name not in ALLOWED:
        errors.append(f'Not allowlisted: {name}')
        continue
    raw = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        errors.append(f'Non-text content: {name}')
        continue
    if len(raw) > 150_000:
        errors.append(f'Unexpected file size: {name}')
    for pattern in (
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
        r'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
        r'\bAKIA[A-Z0-9]{16}\b',
    ):
        if re.search(pattern, text):
            errors.append(f'Credential-pattern match: {name}')
for name in sorted(ALLOWED - tracked):
    errors.append(f'Missing tracked source: {name}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print(f'PASS: {len(tracked)} allowlisted text files; no matched credential patterns.')
print('This is a file-set check, not a proof that all text is non-sensitive.')
