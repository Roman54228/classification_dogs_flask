# Dogs classification with flask :dog2: :scream:

## Clone repo
```
git clone https://github.com/Roman54228/classification_dogs_flask.git
cd classification_dogs_flask/
```

## Download dataset, weights and install dependencies
```
pip install -r requirements.txt
bash download_dataset.sh
```

## Inference on url
```
python inference.py
```
Test on your own example
```
python inference.py --image_url your_url
```

## Run training from scratch

```
python train.py #it will automatically run on downloaded dataset with validation after every epoch
```

```
python train.py --train_path your_path/imagewoof2-320/train --val_path your_path/imagewoof2-320/val --epoch_size number_of_epochs
```

## Run validation on downloaded weights

It will print Accuracy, F1Score, Precision and Recall at the end

```
python val.py
```

```
python val.py --val_path your_path/imagewoof2-320/val
```

## Run flask app with docker 

Simple app to provide inference with web interface, it will look like
![image](snapshot.png)

1) Build image
```
docker build -t pyt .
```

2) Run app

```
docker run -p 127.0.0.1:80:80 --name dogs_flask --rm pyt
```

Now go to 127.0.0.1:80 in your browser and load :sweat_smile:jpg photo!

3) Or enter container and run it directly 
```
docker run --name ubuntu_bash —rm -it dogs_flask bash
python3 app.py
#or
FLASK_ENV=development FLASK_APP=app.py flask run
```



## References :
This app is used for learning purpose and therefore some of the  resources are from : 
- [Pytorch Flask App Tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [Guide to Model deployment](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717)
- [Style CSS background](https://medium.com/@luclemo/styling-background-images-with-css-d9f44cb10a32)
