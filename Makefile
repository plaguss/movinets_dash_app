install:
	pip install -r requirements.txt
lint:
	black app.py
deploy-local:
	python app.py