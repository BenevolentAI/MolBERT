from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

path_to_checkpoint = '/path/to/checkpoint.ckpt'
f = MolBertFeaturizer(path_to_checkpoint)
features, masks = f.transform(['C'])
assert all(masks)
print(features)
