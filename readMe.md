=====================================================================
PROJECT STRUCTURE

data/
    - contains annotated data from doccano

src/
    - contains all code needed to pre-process annotated data
        and train the ner model

finetuned_sample/
    - contains a sample model that can be run
    - check src/train/test.py on how to use it

processed_data/
    - contains checkpoints for each stage in pre-processing
    - this is provided so you don't have to pre-process
        everytime you want to train the model

    **This will be overwritten if you run the pre-processing stage**

=====================================================================

INSTRUCTIONS
**CREATE A VENV THEN INSTALL EVERYTHING IN REQ.TXT**

Pre-processing
    - simply place your annotated jsonl files in **data/raw_jsonl/**
        (it doesn't matter where you put it as long as it's in **data/raw_jsonl/**)
    - make sure you are on the project root
        sample: **/c/Users/Salti/Documents/VS Projects/ner_prep (master)**
    - run as module:
        python -m src.pre_process.pre_proc
    - you should now see each stage of the processed data in **processed_data/**

Training
    - make sure the the last stage of pre_processing is in **processed_data/**
        it is named **stg_3/** by default
    - go to src/train/train.py and change the training arguments as you please
    - run as module:
        python -m src.train.train
    - after training, the checkpoints will be save in **distilbert-finetuned-ner/**
    **The output folder is ignored by git (in .gitignore)   ,**
    **,  as the checkpoints are quite large                 .**
    **,  If you want to share the model you trained         ,**
    **,  simply copy-out the best checkpoint                ,**

Testing
    - go to src/train/test.py and change `checkpoint_path` to your desired model
    - you can also change `sentence` which is the input to be tested
    - run the file
    **A sample model I have finetuned is available at:**
    **https://drive.google.com/file/d/1j0E8B_oitNrUuZY_lluueXLMZvIqNzhe/view?usp=sharing**
    **Download it and extract it in the project root if you want to run it**