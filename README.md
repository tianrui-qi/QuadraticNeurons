This repository contains implementation associated with the 
[paper](https://doi.org/10.1186/s42492-022-00118-z). 
For more detail and future work, please check 
[Fenglei Fan](https://github.com/FengleiFan?tab=stars)'s works and 
[QuadraLib](https://github.com/zarekxu/QuadraLib). 

## Abstract

To enrich the diversity of artificial neurons, a type of quadratic neurons was 
proposed previously, where the inner product of inputs and weights is replaced 
by a quadratic operation. 
In this paper, we demonstrate the superiority of such quadratic neurons over 
conventional counterparts. 
For this purpose, we train such quadratic neural networks using an adapted 
backpropagation algorithm and perform a systematic comparison between quadratic 
and conventional neural networks for classificaiton of Gaussian mixture data, 
which is one of the most important machine learning tasks. 
Our results show that quadratic neural networks enjoy remarkably better efficacy
and efficiency than conventional neural networks in this context, and 
potentially extendable to other relevant applications.

## Keywords

Artifcial neural networks, Quadratic neurons, Quadratic neural networks, 
Backpropagation, Classifcation, Gaussian mixture models

## Conclusions

Although it has been well tested with a solid theoretical foundation, the EM 
algorithm needs to take an entire dataset into the memory, processes them 
iteratively, and is time-consuming, under the restriction that data must come 
from GMM. 
Furthermore, when new samples become available, parameters need to be adjusted 
again. 
A neural network approach can be much more desirable, effective and efficient, 
workable with many data models in principle thanks to its universal 
approximation nature. 
After a network is well trained, new samples can be used to fine-tune the 
network or processed to inference in a feed-forward fashion, being extremely 
efficient and generalizable to cases much more complicated than GMM. 
Very interestingly, compared to conventional networks, quadratic networks can 
deliver a performance close to that of the EM algorithm in the GMM cases and 
yet be orders of magnitude simpler than conventional networks for the same 
classification task.

In conclusion, in this paper we have numerically and experimentally demonstrated
the superiority of quadratic networks over conventional ones. 
It is underlined that the quadratic neural network of a much lighter structure 
rivals the conventional network of a complexity orders of magnitude more in 
solving the same classification problems. 
Clearly, the superior classification performance of quadratic networks could be 
translated to medical imaging tasks, especially radiomics.

## Acknowledgements

This research work was carried out between 2022 and 2023 in 
Dr. [Ge Wang](https://www.linkedin.com/in/ge-wang-axis)'s 
[AI-based X-ray Imaging System (AXIS) Lab](https://wang-axis.github.io/). 
I would like to express my sincere gratitude to Dr. Wang for his guidance and 
assistance. 
His collective expertise has greatly contributed to my research during this 
period.
