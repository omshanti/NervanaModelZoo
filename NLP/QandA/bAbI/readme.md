##Model

This is an implementation of Facebook's baseline GRU/LSTM model on the bAbI dataset 
[Weston et al. 2015](https://research.facebook.com/researchers/1543934539189348).
It includes an interactive demo script [demo.py](./demo.py).

The bAbI dataset contains 20 different question answering tasks.

### Model script

The scripts [train.py](https://github.com/nervanazoo/NervanaModelZoo/blob/master/NLP/QandA/bAbI/train.py)
and [demo.py](https://github.com/nervanazoo/NervanaModelZoo/blob/master/NLP/QandA/bAbI/demo.py) included
here show how to train and run the model, respectively.

### Instructions

These scripts and model weights file are compatible with neon commit [e7ab2c2e2](https://github.com/NervanaSystems/neon/commit/e7ab2c2e27f113a4d36d17ba8c79546faed7d916).

First run the `train.py` script to get a pickle file of model weights. Use the command line arguments `--rlayer_type` 
to choose between LSTMs or GRUs, `--save_path` to specify the output pickle file location, and `-t` to specify which
bAbI task to run.
```
python examples/babi/train.py -e 20 --rlayer_type gru --save_path babi.p -t 15
```

Second run the demo with the newly created pickle file.
```
python examples/babi/demo.py -t 15 --rlayer_type gru --model_weights babi.p
```
```
Task is en/qa15_basic-deduction

Example from test set:

Story
Wolves are afraid of mice.
Sheep are afraid of mice.
Winona is a sheep.
Mice are afraid of cats.
Cats are afraid of wolves.
Jessica is a mouse.
Emily is a cat.
Gertrude is a wolf.

Question
What is emily afraid of?

Answer
wolf

Please enter a story:
```
At which point you can play around with your own stories, questions, and answers.

### Trained weights
The trained weights file for a GRU network trained on task 15 can be downloaded from AWS
using the following link: [trained model weights on task 15][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/bAbI/babi_task15.p

### Performance
Task Number                  | FB LSTM Baseline | Neon QA GRU
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               |  47.9
QA2 - Two Supporting Facts   | 20               |  29.8
QA3 - Three Supporting Facts | 20               |  20.0
QA4 - Two Arg. Relations     | 61               |  69.8
QA5 - Three Arg. Relations   | 70               |  56.4
QA6 - Yes/No Questions       | 48               |  49.1
QA7 - Counting               | 49               |  76.5
QA8 - Lists/Sets             | 45               |  68.9
QA9 - Simple Negation        | 64               |  62.8
QA10 - Indefinite Knowledge  | 44               |  45.3
QA11 - Basic Coreference     | 72               |  67.6
QA12 - Conjunction           | 74               |  63.9
QA13 - Compound Coreference  | 94               |  91.9
QA14 - Time Reasoning        | 27               |  36.8
QA15 - Basic Deduction       | 21               |  51.4
QA16 - Basic Induction       | 23               |  50.1
QA17 - Positional Reasoning  | 51               |  49.0
QA18 - Size Reasoning        | 52               |  90.5
QA19 - Path Finding          | 8                |   9.0
QA20 - Agent's Motivations   | 91               |  95.6

## Citation

[Facebook research](https://research.facebook.com/researchers/1543934539189348)

```
Weston, Jason, et al. "Towards AI-complete question answering: a set of prerequisite toy tasks." arXiv preprint arXiv:1502.05698 (2015).
```
