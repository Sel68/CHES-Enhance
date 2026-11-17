import json
import os
import subprocess
import urllib.request

with urllib.request.urlopen('https://pypi.org/pypi/jax/json', timeout=20) as response:
    metadata = json.load(response)
url = next(item['url'] for item in metadata['urls'] if item['filename'].endswith('.whl'))
for options in (['-4'], ['-4', '--http1.1']):
    command = ['curl', *options, '-L', '--range', '0-1048575', '--max-time', '20', '-o', os.devnull, '-w', 'bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n', url]
    print('Testing', options, flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout, result.stderr[-300:], flush=True)
