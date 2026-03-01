python run.py --config configs/demo/demo-0.yaml
python run.py --postprocess data/results/satellite-b/demo-0
python run.py --postprocess data/results/satellite-b/demo-0/noise-multiplier-0
python run.py --job noise-multiplier-0 --postprocess data/results/satellite-b/demo-0

python run.py --config configs/satellite-b/test-parallel.yaml
python run.py --postprocess data/results/satellite-b/test-parallel
python run.py --postprocess data/results/satellite-b/test-parallel/cluster1
python run.py --job cluster1 --postprocess data/results/satellite-b/test-parallel

python run.py --config configs/satellite-b/test-serial.yaml
python run.py --postprocess data/results/satellite-b/test-serial
python run.py --postprocess data/results/satellite-b/test-serial/cluster1
python run.py --job cluster1 --postprocess data/results/satellite-b/test-serial
