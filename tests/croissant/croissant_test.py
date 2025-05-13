import mlcroissant as mlc


ds = mlc.Dataset("https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/1.0/gpt-3/metadata.json")
metadata = ds.metadata.to_json()
print(f"{metadata['name']}: {metadata['description']}")
for x in ds.records(record_set="default"):
    print(x)
