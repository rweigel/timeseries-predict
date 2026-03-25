import numpy as np

def stats(reps):
  arvs = {}
  for i, rep in enumerate(reps):
    models = list(rep['models'].keys())
    for model in models:
      if model not in arvs:
        arvs[model] = {
          'train': np.empty((len(reps), len(rep['outputs']))),
          'test': np.empty((len(reps), len(rep['outputs'])))
        }
        if 'train*' in rep['models'][model]['metrics']:
          arvs[model]['train*'] = np.empty((len(reps), len(rep['outputs'])))
        if 'test*' in rep['models'][model]['metrics']:
          arvs[model]['test*'] = np.empty((len(reps), len(rep['outputs'])))

      arvs[model]['train'][i, :] = rep['models'][model]['metrics']['train']
      arvs[model]['test'][i, :] = rep['models'][model]['metrics']['test']

      if 'train*' in rep['models'][model]['metrics']:
        arvs[model]['train*'][i, :] = rep['models'][model]['metrics']['train*']
      if 'test*' in rep['models'][model]['metrics']:
        arvs[model]['test*'][i, :] = rep['models'][model]['metrics']['test*']

  stats = {}
  for i, output in enumerate(reps[0]['outputs']):
    stats[output] = {}
    for model in arvs:
      stats[output][model] = {'train': {}, 'test': {}}
      arvs_train = arvs[model]['train'][:, i]
      arvs_test = arvs[model]['test'][:, i]
      stats[output][model]['train'] = {'mean': np.mean(arvs_train), 'se': None}
      stats[output][model]['test'] = {'mean': np.mean(arvs_test), 'se': None}
      if 'train*' in arvs[model]:
        arvs_train_star = arvs[model]['train*'][:, i]
        stats[output][model]['train*'] = {'mean': np.mean(arvs_train_star), 'se': None}
      if 'test*' in arvs[model]:
        arvs_test_star = arvs[model]['test*'][:, i]
        stats[output][model]['test*'] = {'mean': np.mean(arvs_test_star), 'se': None}

      n = arvs[model]['train'].shape[0]
      if n > 19:
        # Only compute standard error used for uncertainty if number of repetitions > 19
        stats[output][model]['train']['se'] = np.std(arvs_train, ddof=1)/np.sqrt(n)
        stats[output][model]['test']['se'] = np.std(arvs_test, ddof=1)/np.sqrt(n)
        if 'train*' in arvs[model]:
          stats[output][model]['train*']['se'] = np.std(arvs_train_star, ddof=1)/np.sqrt(n)
        if 'test*' in arvs[model]:
          stats[output][model]['test*']['se'] = np.std(arvs_test_star, ddof=1)/np.sqrt(n)

  return stats