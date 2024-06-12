# YusufDagdeviren/SentimentAnalysisFromMovieReviews

This model is a fine-tuned version of [xlnet-base-cased](https://huggingface.co/xlnet-base-cased) on the imdb dataset.
It achieves the following results on the evaluation set:
- Loss: 0.16
- Accuracy: 0.93
- F1: 0.93

## Model Description

This project uses a fine-tuned XLNet model for sentiment analysis on English movie reviews. The model was fine-tuned using PyTorch and Huggingface Transformers libraries to improve its performance on sentiment classification tasks.

XLNet (eXtreme Language Model) is an autoregressive pre-training method that combines the best of BERT and Transformer-XL architectures, providing significant improvements in performance over traditional language models. This fine-tuned XLNet model aims to provide high accuracy and reliability in sentiment analysis.

The training process involved the use of the AdamW optimizer with a learning rate of 2e-5, betas of [0.9, 0.999], and epsilon of 1e-6. The model was trained for 2 epochs with a linear learning rate scheduler and no warmup steps.


## Training and Evaluation Data

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


### Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-5
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- total_train_batch_size: 38
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-6
- lr_scheduler_type: linear
- num_epochs: 2
### Training Results

======== Epoch 1 / 2 ========  
Training...  
  Batch    30  of  1,222.    Elapsed: 0:00:38.  
  Batch    60  of  1,222.    Elapsed: 0:01:16.  
  Batch    90  of  1,222.    Elapsed: 0:01:53.  
  Batch   120  of  1,222.    Elapsed: 0:02:30.  
  Batch   150  of  1,222.    Elapsed: 0:03:07.  
  Batch   180  of  1,222.    Elapsed: 0:03:44.  
  Batch   210  of  1,222.    Elapsed: 0:04:21.  
  Batch   240  of  1,222.    Elapsed: 0:04:58.  
  Batch   270  of  1,222.    Elapsed: 0:05:35.  
  Batch   300  of  1,222.    Elapsed: 0:06:12.  
  Batch   330  of  1,222.    Elapsed: 0:06:49.  
  Batch   360  of  1,222.    Elapsed: 0:07:27.  
  Batch   390  of  1,222.    Elapsed: 0:08:04.  
  Batch   420  of  1,222.    Elapsed: 0:08:41.  
  Batch   450  of  1,222.    Elapsed: 0:09:18.  
  Batch   480  of  1,222.    Elapsed: 0:09:55.  
  Batch   510  of  1,222.    Elapsed: 0:10:32.  
  Batch   540  of  1,222.    Elapsed: 0:11:09.  
  Batch   570  of  1,222.    Elapsed: 0:11:46.  
  Batch   600  of  1,222.    Elapsed: 0:12:24.  
  Batch   630  of  1,222.    Elapsed: 0:13:01.  
  Batch   660  of  1,222.    Elapsed: 0:13:38.  
  Batch   690  of  1,222.    Elapsed: 0:14:15.  
  Batch   720  of  1,222.    Elapsed: 0:14:52.  
  Batch   750  of  1,222.    Elapsed: 0:15:29.  
  Batch   780  of  1,222.    Elapsed: 0:16:06.  
  Batch   810  of  1,222.    Elapsed: 0:16:43.  
  Batch   840  of  1,222.    Elapsed: 0:17:20.  
  Batch   870  of  1,222.    Elapsed: 0:17:57.  
  Batch   900  of  1,222.    Elapsed: 0:18:35.  
  Batch   930  of  1,222.    Elapsed: 0:19:12.  
  Batch   960  of  1,222.    Elapsed: 0:19:49.  
  Batch   990  of  1,222.    Elapsed: 0:20:26.  
  Batch 1,020  of  1,222.    Elapsed: 0:21:03.  
  Batch 1,050  of  1,222.    Elapsed: 0:21:40.  
  Batch 1,080  of  1,222.    Elapsed: 0:22:17.  
  Batch 1,110  of  1,222.    Elapsed: 0:22:54.  
  Batch 1,140  of  1,222.    Elapsed: 0:23:31.  
  Batch 1,170  of  1,222.    Elapsed: 0:24:09.  
  Batch 1,200  of  1,222.    Elapsed: 0:24:46.  

  Average training loss: 0.27  
  Training epoch took: 0:25:12  

Running Validation...  
  Accuracy: 0.92  
  Validation took: 0:02:51  

======== Epoch 2 / 2 ========  
Training...  
  Batch    30  of  1,222.    Elapsed: 0:00:37.  
  Batch    60  of  1,222.    Elapsed: 0:01:14.  
  Batch    90  of  1,222.    Elapsed: 0:01:51.  
  Batch   120  of  1,222.    Elapsed: 0:02:29.  
  Batch   150  of  1,222.    Elapsed: 0:03:06.  
  Batch   180  of  1,222.    Elapsed: 0:03:43.  
  Batch   210  of  1,222.    Elapsed: 0:04:20.  
  Batch   240  of  1,222.    Elapsed: 0:04:57.  
  Batch   270  of  1,222.    Elapsed: 0:05:34.  
  Batch   300  of  1,222.    Elapsed: 0:06:11.  
  Batch   330  of  1,222.    Elapsed: 0:06:48.  
  Batch   360  of  1,222.    Elapsed: 0:07:25.  
  Batch   390  of  1,222.    Elapsed: 0:08:03.  
  Batch   420  of  1,222.    Elapsed: 0:08:40.  
  Batch   450  of  1,222.    Elapsed: 0:09:17.  
  Batch   480  of  1,222.    Elapsed: 0:09:54.  
  Batch   510  of  1,222.    Elapsed: 0:10:31.  
  Batch   540  of  1,222.    Elapsed: 0:11:08.  
  Batch   570  of  1,222.    Elapsed: 0:11:45.  
  Batch   600  of  1,222.    Elapsed: 0:12:23.  
  Batch   630  of  1,222.    Elapsed: 0:13:00.  
  Batch   660  of  1,222.    Elapsed: 0:13:37.  
  Batch   690  of  1,222.    Elapsed: 0:14:14.  
  Batch   720  of  1,222.    Elapsed: 0:14:51.  
  Batch   750  of  1,222.    Elapsed: 0:15:28.  
  Batch   780  of  1,222.    Elapsed: 0:16:05.  
  Batch   810  of  1,222.    Elapsed: 0:16:43.  
  Batch   840  of  1,222.    Elapsed: 0:17:20.  
  Batch   870  of  1,222.    Elapsed: 0:17:57.  
  Batch   900  of  1,222.    Elapsed: 0:18:34.  
  Batch   930  of  1,222.    Elapsed: 0:19:11.  
  Batch   960  of  1,222.    Elapsed: 0:19:48.  
  Batch   990  of  1,222.    Elapsed: 0:20:25.  
  Batch 1,020  of  1,222.    Elapsed: 0:21:03.  
  Batch 1,050  of  1,222.    Elapsed: 0:21:40.  
  Batch 1,080  of  1,222.    Elapsed: 0:22:17.  
  Batch 1,110  of  1,222.    Elapsed: 0:22:54.  
  Batch 1,140  of  1,222.    Elapsed: 0:23:31.  
  Batch 1,170  of  1,222.    Elapsed: 0:24:08.  
  Batch 1,200  of  1,222.    Elapsed: 0:24:45.  

  Average training loss: 0.16  
  Training epoch took: 0:25:12  

Running Validation...  
  Accuracy: 0.93  
  Validation took: 0:02:52  

### Framework Versions
- Transformers 4.41.2  
- Pytorch 2.3  
- Tokenizers 0.19.1  