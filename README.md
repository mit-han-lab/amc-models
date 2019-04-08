# AMC Compressed Models

This repo contains some of the compressed models from paper **AMC: AutoML for Model Compression and Acceleration on Mobile Devices (ECCV18)**.

## Reference

If you find the models useful, please kindly cite our paper:

```
@inproceedings{he2018amc,
  title={Amc: Automl for model compression and acceleration on mobile devices},
  author={He, Yihui and Lin, Ji and Liu, Zhijian and Wang, Hanrui and Li, Li-Jia and Han, Song},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={784--800},
  year={2018}
}
```

## Download the Pretrained Models

Firstly, download the pretrained models from [here](https://drive.google.com/drive/folders/1w_1OyKuoj8JciwKPFNvztSmAqEEwPFlA?usp=sharing) and put it in `./checkpoints`.

## Models

### Compressed MobileNets

We provide compressed MobileNetV1 by **50% FLOPs** and **50% Inference time**, and also compressed MobileNetV2 by **70% FLOPs**, with PyTorch. The comparison with vanila models as follows:

| Models                   | Top1 Acc (%) | Top5 Acc (%) | Latency (ms) | MACs (M) |
| ------------------------ | ------------ | ------------ | ------------ | -------- |
| MobileNetV1              | 70.9         | 89.5         | 123          | 569      |
| MobileNetV1-width*0.75   | 68.4         | 88.2         | 72.5         | 325      |
| **MobileNetV1-50%FLOPs** | **70.5**     | **89.3**     | 68.9         | 285      |
| **MobileNetV1-50%Time**  | **70.2**     | **89.4**     | 63.2         | 272      |
| MobileNetV2-width*0.75   | 69.8         | 89.6         | -            | 300      |
| **MobileNetV2-70%FLOPs** | **70.9**     | **89.9**     | -            | 210      |

To test the model, run:

```
python eval_mobilenet_torch.py --profile={mobilenet_0.5flops, mobilenet_0.5time, mobilenetv2_0.7flops}
```



### Converted TensorFLow Models

We converted the **50% FLOPs** and **50% time** compressed MobileNetV1 model to TensorFlow. We offer the normal checkpoint format and also the TF-Lite format. We used the TF-Lite format to test the speed on MobileNet.

To replicate the results of PyTorch, we write a new preprocessing function, and also adapt some hyper-parameters from the original TF [MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md). To verify the performance, run following scripts:

```
python eval_mobilenet_tf.py --profile={0.5flops, 0.5time}
```

The produced result is:

| Models    | Top1 Acc (%) | Top5 Acc (%) |
| --------- | ------------ | ------------ |
| 50% FLOPs | 70.424       | 89.28        |
| 50% Time  | 70.214       | 89.244       |

## Timing Logs

Here we provide timing logs on Google Pixel 1 using **TensorFlow Lite** in `./logs` directory. We benchmarked the original MobileNetV1 (mobilenet), MobileNetV1 with 0.75 width multiplier (0.75mobilenet), 50% FLOPs pruned MobileNetV1 (0.5flops) and 50% time pruned MobileNetV1 (0.5time). Each model is benchmarked for 200 iterations with extra 100 iterations for warming up, and repeated for 3 runs. 

## Contact

To contact the authors:

Ji Lin, jilin@mit.edu

Song Han, songhan@mit.edu