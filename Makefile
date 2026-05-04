rsync-to-mag:
	rsync -avz --delete data/ weigel@mag.gmu.edu:www/git-data/timeseries-predict/data/

rsync-from-mag:
	rsync -avz weigel@mag.gmu.edu:www/git-data/timeseries-predict//data ./data/