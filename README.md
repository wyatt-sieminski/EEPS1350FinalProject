# EEPS 1350 Final Project Reproducibility and Data Availability

This is the final project repository for Wyatt Sieminski's Final project downscaling nighttime irradiance data using a CNN. 

Environment: A python environment must be setup using all of the packages listed in requirements.txt

Training and running the model: The model can be run and training using the `python main.py` command in a terminal with the environment active. Editing the `main()` function in `main.py` controls which aspects of the modeling process are active. The model can be trained by running the line `model = train_model(training_dataloader, testing_dataloader)`. The model can then be saved locally, and loaded back in locally to avoid retraining the model whenever running analysis code. This can all be done by adjusting which lines within `main.py` run. 

Data: Data for roads and population density can be found at https://www.census.gov/cgi-bin/geo/shapefiles/index.php, and are taken for Rhode Island from the year 2024. VIIRS irradiance data can be found at https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A2/2024/, the first 10 days of 2024 are used in this work. The h10v4 files for each day are downloaded and the file names are simplified to be read in by the code in `utils.py'.