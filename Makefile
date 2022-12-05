#################### PACKAGE ACTIONS #################

reinstall:
	@pip uninstall -y youtube || :
	@pip install -e .

#################### MODEL ACTIONS ###################

preprocess:
	python -c 'from youtube.interface.main import preprocess; preprocess()'

train:
	python -c 'from youtube.interface.main import train; train()'

evaluate:
	python -c 'from youtube.interface.main import train; train(evaluate=True)'

predict:
	python -c 'from youtube.interface.main import pred; pred()'

all: preprocess train pred evaluate

#################### API ACTIONS ###################

api:
	uvicorn youtube.fast_api.fast:app --reload
