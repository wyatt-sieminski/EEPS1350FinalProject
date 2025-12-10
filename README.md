# EEPS 1350 Final Project Reproducibility and Data Availability

This is the final project repository for Wyatt Sieminski's Final project downscaling nighttime irradiance data using a CNN. 

Environment: A python environment must be setup using all of the packages listed in requirements.txt

Training and running the model: The model can be run and training using the `python main.py` command in a terminal with the environment active. Editing the `main()` function in `main.py` controls which aspects of the modeling process are active. The model can be trained by running the line `model = train_model(training_dataloader, testing_dataloader)`. The model can then be saved locally, and loaded back in locally to avoid retraining the model whenever running analysis code. This can all be done by adjusting which lines within `main.py` run. 

Data: 