## Sunday, 3 January 2023
- from a survey on quantization (https://arxiv.org/pdf/2103.13630.pdf), I found a paper that focuses on QAT using a "quantization function" with learnable parameters that allows different choices of bitwidth through some hyperparameters of the same function: (https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf).

## Sunday, 8 January 2023
- from Qualcomm AI Research I found a recent (jun 2021) and interesting white paper (https://arxiv.org/pdf/2106.08295.pdf) on best-practices in PTQ and QAT pipelines. From the same group, another research suggests FP8 instead of INT8 on PTQ especially to capture outliers, while in QAT, INT8 remains the best choice: (https://arxiv.org/pdf/2208.09225.pdf).
- found ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)

next steps:
 - try to implement Quantization Networks for your super-resolution UNet.
 - refer to (https://github.com/aliyun/alibabacloud-quantization-networks)
 - read white paper by Qualcomm

## Monday, 16 January 2023
- today I've done nothing, last week I did nothing: I think it's time to organize my work in a different way
- I should buy a fucking desk, and do at least a deep-work block per day

## Thursday, 19 January 2023
- I've started studying alibabacloud-quantization-networks repo
- I should need another couple of sessions to read the code, then I should be able to start implementing something for the UNet on my own

## Tuesday, 7 February 2023
- too slow bruh, need to speed up
- I'm going to reimplement our current UNet architecture using XNOR-Net blocks, and its training/evaluation procedures

## Tuesday, 14 February 2023
- I've changed my mind, now I'm going to test if I can implement a PTQ on UNet using the Torch-TensorRT framework

## Tuesday, 22 February 2023
- Compiled UNet and SRUNet with `pytorch_tensorrt` for inference
- Tested speed up gain given by compiled models
- I think we're ready to start writing bruh
- Only thing that misses is to understand a bit better tensorrt under the hood
