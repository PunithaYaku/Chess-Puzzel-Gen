.PHONY: setup test run train clean

setup:
	python -m venv venv
	.\venv\Scripts\activate && pip install -r requirements.txt

test:
	.\venv\Scripts\activate && pytest tests/

run:
	.\venv\Scripts\activate && python app.py

train:
	.\venv\Scripts\activate && python train_gen.py

clean:
	rmdir /s /q __pycache__
	del /q *.pth
