import json

data_file = "/workspaces/StyleAdv-CDFSL/filelists/miniImagenet/base.json"

with open(data_file, 'r') as f:
    meta = json.load(f)

print(meta)