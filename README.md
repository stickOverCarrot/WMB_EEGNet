# Motor imagery EEG decoding based on weight multi-branch structure suitable for multi-subject data 
## Environment
* python 3.7
* pytorch 1.8.0
* braindecode package is directly copied from https://github.com/robintibor/braindecode/tree/master/braindecode for preparing datasets 
## Start
* setp 1 Prepare dataset(Only needs to run once)
   
    `python data/data_bciciv2a_tools.py --data_path ~/dataset/bciciv2a/gdf -output_path ~/dataset/bciciv2a/pkl`
* step 2 Train model 
  
    `python main.py -data_path ~/dataset/bciciv2a/pkl -target_id 1`
## Licence
For academtic and non-commercial usage
