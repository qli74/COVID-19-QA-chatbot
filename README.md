# COVID-19 QA chat bot 
This project is based on ParlAI framework.

code link: https://github.com/qli74/ParlAI

### 1.Download files and check the data
```
git clone https://github.com/qli74/ParlAI
cd ParlAI
python setup.py install
python examples/display_data.py -t covid -n 1
```

### 2.Train a model 

key parameters:
* --batchsize: training batchsize
* -ttim: training time
* -stim: time interval of saving checkpoint
* --model-file: model path
```
python examples/train_model.py --init-model zoo:pretrained_transformers/model_poly/model -t covid \
  --model transformer/polyencoder --batchsize 5 --eval-batchsize 10 \
  --warmup_updates 100 --optimizer admax --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 2e-03 --data-parallel True --history-size 20 --label-truncate 100 \
  --text-truncate 360 --num-epochs 20.0 --max_train_time 200000 -veps -1 \
  -vme 8000 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid False --log_every_n_secs 20 --candidates batch --fp16 True \
  --dict-tokenizer bpe --dict-lower True --output-scaling 0.06 \
  --variant xlm --reduction-type mean --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
  --poly-attention-type basic --dict-endtoken __start__ \
  --dict-file  data/models/pretrained_transformers/model_poly/model.dict \
  --model-file ../models/covid -ttim 20000 -stim 200
```

### 3. Evaluate the model
```
python examples/eval_model.py -m transformer/polyencoder -mf ../models/covid -t covid --encode-candidate-vecs true --eval-candidates inline
```

### 4. Interact
```
python examples/interactive.py -m transformer/polyencoder -mf ../models/covid --encode-candidate-vecs true --single-turn True
```

### 5.web chat
```
./start_browser_service.sh
```
Port number (default: 9924) is set in start_browser_service.sh

IP address (default: 0.0.0.0) is set in PariAI/parlai/chat_service/services/browser_chat/client.py _run_browser()

![example](https://github.com/qli74/ParlAI/blob/master/cov1.png)

### 6.terminal chat
```
./start_terminal_service.sh
```
![example](https://github.com/qli74/ParlAI/blob/master/cov2.png)


### 7.another api file written with fastapi: ParlAI/fastapi_covid.py\
https://github.com/qli74/ParlAI/blob/master/fastapi_covid.py
