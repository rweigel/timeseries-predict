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

      arvs[model]['train'][i,:] = rep['models'][model]['metrics']['train']
      arvs[model]['test'][i,:] = rep['models'][model]['metrics']['test']

  stats = {}
  for i, output in enumerate(reps[0]['outputs']):
    stats[output] = {}
    for model in arvs:
      stats[output][model] = {'train': {}, 'test': {}}
      arvs_train = arvs[model]['train'][:, i]
      arvs_test = arvs[model]['test'][:, i]
      stats[output][model]['train']['mean'] = np.mean(arvs_train)
      stats[output][model]['test']['mean'] = np.mean(arvs_test)
      stats[output][model]['test']['se'] = None
      stats[output][model]['train']['se'] = None
      n = arvs[model]['train'].shape[0]
      if n > 19:
        # Only compute standard error used for uncertainty if number of repetitions > 19
        stats[output][model]['train']['se'] = np.std(arvs_train, ddof=1)/np.sqrt(n)
        stats[output][model]['test']['se'] = np.std(arvs_test, ddof=1)/np.sqrt(n)

  return stats