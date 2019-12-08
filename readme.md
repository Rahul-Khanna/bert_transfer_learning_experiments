Rahul Khanna

rahulkha@usc.edu

USC ID: 1599870732

NLP Project -- Exploring BERT's Transfer Learning abilities on the TabFact Dataset

Steps To Reproduce Results:

Replicating Baseline Results:

1. git clone https://github.com/wenhuchen/Table-Fact-Checking.git
2. Set up virtual environment with python 3.5
3. Start virtual environmnent
4. `pip install -r replication_rqs.txt`
5. cd Table-Fact-Checking/code
6. `python run_BERT.py --do_train --do_eval --scan horizontal --fact first`

Replicating Frozen TableBERT Results:

1. git clone this repo
2. Set up virtual environment with python 3.5
3. Start virtual environment
4. `pip install -r rqs.txt`
5. cd into clone repo
6. `mkdir default_features`

Experiment Number 1:

    python3.5 run_bert_rahul.py --do_train --scan horizontal --fact first --eval_batch_size=128 --model_type=bert --model_name=bert-base-multilingual-cased --period=100 --train_batch_size=128 --output_dir=outputs --load_cached_features --learning_rate=0.03368973499

Experiment Number 2:

    python3.5 run_bert_rahul.py --do_train --scan horizontal --fact first --eval_batch_size=64 --model_type=bert --model_name=bert-base-multilingual-cased --period=300 --train_batch_size=64 --output_dir=outputs --load_cached_features --learning_rate=0.01347589399

in `outputs_fact-first_horizontal` you will find a file called `eval_metrics.txt` which will have the evaluation results you are looking for.

You can also view the report in this github, it is Report.ipynb

This code is heavily reliant on the code found here: https://github.com/wenhuchen/Table-Fact-Checking.git

The data in `processed_datasets` was taken from the above repo.
