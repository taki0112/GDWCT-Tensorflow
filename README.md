<img src="https://user-images.githubusercontent.com/20943085/59210582-e27c9600-8be8-11e9-8434-148cc3bdb274.png" width="100%"></img>


### [Paper](https://arxiv.org/abs/1812.09912) | [Pytorch code](https://github.com/WonwoongCho/GDWCT)

# Will be soon

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
           
├── guide.jpg (example for guided image translation task)
```

### Train
```
 > python main.py --dataset male2female
```

### Test
```
 > python main.py --dataset male2female --phase test
```

## Results

## Author
Junho Kim
