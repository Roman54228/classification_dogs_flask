FROM python:3

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN gdown https://drive.google.com/uc?id=1-0nYK87buY7c3bys9R3NWp7NPDTwzgf9

COPY . .

CMD ["python", "app.py"]
