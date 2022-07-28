# Dogs classification with flask :dog2: :scream:

## Description

Model: ConvNeXtBase, 89M params, 15.4G FLOPS, acc=85.8% on ImageNet1k
Was training only last fc layer, used pretrained weights from torchvision. Model is too large for full training on colab GPU.

## Metrics

| Accuracy      | F1Score      |   
| ------------- | ------------- | 
| 92% | 92% |
  



## Clone repo
```
git clone https://github.com/Roman54228/classification_dogs_flask.git
cd classification_dogs_flask
```

## Download dataset, weights and install dependencies
```
pip install -r requirements.txt
bash download_dataset.sh
bash download_weights.sh
```


Simple app to provide inference with web interface, it will look like
![image](snapshot.png)

## View in browser with Flask
```
python3 app.py
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
python train.py 
#it will automatically run on downloaded dataset with validation after every epoch
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


1) Build image
```
docker build -t pyt .
```

2) Run app

```
docker run -p 127.0.0.1:80:80 --name dogs_flask --rm pyt
```

Now go to 127.0.0.1:80 in your browser and load jpg photo!

3) Or enter container and run it directly 
```
docker run --name ubuntu_bash —rm -it dogs_flask bash
python3 app.py
#or
FLASK_ENV=development FLASK_APP=app.py flask run
```



## References :
This app is used for learning purpose and therefore some of the  resources are from : 
- [ConvNeXt official repo](https://github.com/facebookresearch/ConvNeXt)
- [ConvNeXt paper](https://arxiv.org/pdf/2201.03545v2.pdf)

