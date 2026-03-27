
import json
import re

path = 'podcast_studio_CPU.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the broken lines by identifying the raw newline before \",
# It seems there's a literal newline in the middle of a JSON string.
# We want to replace something like: 'pyrootutils>=1.0.4',\n\", with 'pyrootutils>=1.0.4',\\n\",
# Note the double backslash in the replacement to get a literal \n in the file.

text = re.sub(r"'pyrootutils>=1.0.4',\n\s*\",", r"'pyrootutils>=1.0.4',\\n\",", text, flags=re.MULTILINE)
text = re.sub(r"'psutil',\n\s*\",", r"'psutil',\\n\",", text, flags=re.MULTILINE)
text = re.sub(r"'reimplemented==0.1.7',\n\s*\",", r"'reimplemented==0.1.7',\\n\",", text, flags=re.MULTILINE)

# Also check for bfloat16 problem from earlier shell attempt
text = text.replace('`bfloat16`\\n\",', '`bfloat16`.\\n\",')
# (I think I added an extra quote earlier as well)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

# Now try to load as JSON to verify
try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("SUCCESS: JSON is valid.")
except json.JSONDecodeError as e:
    print(f"FAILED: {e}")
    # Print the offending line
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if e.lineno <= len(lines):
            print(f"Line {e.lineno}: {repr(lines[e.lineno-1])}")
