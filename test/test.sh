python run.py --config configs/demo/demo-0.yaml
python run.py --postprocess data/results/demo/demo-0
python run.py --postprocess data/results/demo/demo-0/noise-multiplier-0
python run.py --job noise-multiplier-0 --postprocess data/results/demo/demo-0

python run.py --config configs/satellite-b/parallel-test.yaml
python run.py --postprocess data/results/satellite-b/parallel-test
python run.py --postprocess data/results/satellite-b/parallel-test/cluster1
python run.py --job cluster1 --postprocess data/results/satellite-b/parallel-test

python run.py --config configs/satellite-b/serial-test.yaml
python run.py --postprocess data/results/satellite-b/serial-test
python run.py --postprocess data/results/satellite-b/serial-test/cluster1
python run.py --job cluster1 --postprocess data/results/satellite-b/serial-test
