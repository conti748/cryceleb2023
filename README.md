
# cryceleb2023: Triplet Loss for Infant Cry Verification - CryCeleb2023 Solution

This repository presents a solution to the problem of verifying audio
tracks containing crying babies, as proposed in the [CryCeleb2023](https://huggingface.co/spaces/competitions/CryCeleb2023) challenge.
The proposed solution achieved the 3rd place in the private leadebord and the 1st
in the public leaderbord.
The solution builds upon the [baseline](https://github.com/Ubenwa/cryceleb2023) approach provided in the notebook 
accompanying the challange. 
In details, while the baseline utilizes transfer learning by fine-tuning a classifier and extracting embeddings without the classification head, the herein presented approach fine-tune the same network using triplet-loss, enabling the model to directly learn an embedding representation.

You can find:
- evaluation: a notebook to evaluate the performance of the trained network and reproduce the performance achieved in the competition on the dev and test sets.
- training: a notebook to reproduce the training of the network, allowing you to fine-tune the model using triplet-loss.
- technical report: a technical discussion detailing the approach used in this solution, providing in-depth insights into the methodology and techniques employed.

  
By exploring the provided notebooks and technical report, you can gain a comprehensive understanding of the proposed solution and further contribute to advancements in audio analysis and verification.



## Technical Report

### CryCeleb challenge

The evaluation set consists of pairs of baby cry recordings from the time of birth and the time of discharge from the hospital. The task is to predict if both pairs come from the same baby.

### CryCeleb base-line approach

XXXXXX

### Proposed Solution

The proposed solution introduces the following key improvements over the baseline approach:

1) Training an embedding with Triplet Loss:

Instead of training a classifier, the solution focuses on training an 
embedding using a triplet loss approach. The primary objective is to 
minimize the embedding distance between samples from the same babies
while simultaneously maximizing the distance for pairs of different infants.
By adopting a training loop based on online hard batch mining, the solution
dynamically generates pairs of samples to ensure effective discrimination
between similar and dissimilar instances. This approach facilitates 
learning a more discriminative embedding space, which can enhance the accuracy of the subsequent verification task.

2) Unified Dataset Split Technique:

Unlike the baseline approach, which distinguishes between the 'B' and 'D' 
periods of audio track capture, the solution considers the entire dataset 
for the verification task. From experiments, attempts to divide the dataset 
into training and validation sets using different split techniques 
did not yield successful results. Therefore, a unified
approach is used to leverage the complete dataset for training the embedding. 
By not separating the audio tracks based on periods,
the solution maximizes the available data for training, 
potentially compensating for the limited number of samples 
in the challenge dataset.

By employing the triplet loss training and the unified dataset split 
technique, our solution offers several advantages over the baseline approach:

- Enhanced Representation Learning: The use of triplet loss training enables
the network to learn more nuanced representations, capturing subtle
differences between crying babies. This improved representation learning
can enhance the model's ability to discriminate between different 
instances effectively. Further, It is widely shared that triplet loss performs
better than Additive Angular Margin (used in the baseline) on small-sized datasets.

- Focus on Challenging Samples: The adoption of online hard batch mining
ensures that the network focuses on the most challenging samples during
training. By prioritizing difficult samples, the model further improves
its discriminative capabilities, leading to enhanced performance.

- Maximizing Data Utilization: By utilizing the complete dataset without 
splitting based on periods, our solution maximizes the available data for
training the embedding. This approach potentially mitigates the limitations
posed by a limited number of samples, allowing the model to generalize
better.

Overall, the modifications enhance the solution's ability to tackle the problem of
verifying audio tracks containing crying babies. These improvements can lead to higher 
accuracy and better generalization compared to the baseline approach.

### Remark
It is emphasized that a proper hyperparameter selection was not performed due to limited resource availability. In fact, all trainings were conducted using Google Colab. This limitation may significantly impact the performance and further improvements can be achieved by conducting a thorough hyperparameter tuning.


The entire solution has been developed using Google Colab without a proper
hyper-parameter selection, that can further improve the performances.



## Evaluation
The notebook evaluate.ipynb evaluate the prediction using the fine-tuned model on the dev-set and the test-set to produce the final submission.
The notebook download the data from HF_hub and the fine-tuned model from Google-Drive, you can manually download the model [here](https://drive.google.com/file/d/1eZnYIlL5ZrLKoqBoEUow9M_EfX1Xt0MQ/view?usp=sharing).

Here, you can find the performance on the dev-set.... XXXXX


## Training

To train the model, follow these steps:
