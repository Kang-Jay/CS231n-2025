# CS231n(Spring 2025) Solutions & Study Notes

This repository collects study materials for Stanford CS231n — Deep Learning for Computer Vision (Spring 2025). It includes my personally compiled and implemented assignment reference code, chapter-by-chapter study notes, and model diagrams visualized in the notes. Note: these materials are my personal compilations and implementations, not official solutions; they are provided for learning and reference only. You are welcome to discuss issues and improvements via GitHub issues or the CSDN comments section; for private contact, you may email [kang-jay@qq.com](mailto:kang-jay@qq.com).

## Course Resources

Official course website: [Stanford University CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)

Assignments: [CS231n Assignments](https://cs231n.stanford.edu/assignments.html)

Assignment source archives:

- Assignment1: <https://cs231n.github.io/assignments/2025/assignment1_colab.zip>
- Assignment2: <https://cs231n.github.io/assignments/2025/assignment2_colab.zip>
- Assignment3: <https://www.mediafire.com/file/az17sl7q7eroxi2/assignment3.zip/file>

Slides/schedule: <https://cs231n.stanford.edu/schedule.html>

2025 Lecture videos:

- Bilibili: <https://www.bilibili.com/video/BV1b1agz5ERC>
- Youtube: <https://youtu.be/2fq9wYslV0A>

You can also download the compiled course resources (source code for the three assignments, all lecture PPTs, and several model diagrams used in the notes): <https://www.mediafire.com/file/1mhnbhu129o82vl/Resources.zip/file>

## Notes

| Assignment  | Chapter | Topic                              | Notes                                                        |
| ----------- | ------- | ---------------------------------- | ------------------------------------------------------------ |
| Assignment1 | 1-1     | kNN                                | <https://blog.csdn.net/x2114754480/article/details/149572662> |
|             | 1-2     | Softmax                            | <https://blog.csdn.net/x2114754480/article/details/149689949> |
|             | 1-3     | Two-Layer Neural Network           | <https://blog.csdn.net/x2114754480/article/details/149866392> |
|             | 1-4     | Image Features                     | <https://blog.csdn.net/x2114754480/article/details/152214887> |
|             | 1-5     | Fully-Connected Neural Network     | <https://blog.csdn.net/x2114754480/article/details/149941584> |
| Assignment2 | 2-1     | Batch Normalization                | <https://blog.csdn.net/x2114754480/article/details/150061156> |
|             | 2-2     | Dropout                            | <https://blog.csdn.net/x2114754480/article/details/150119299> |
|             | 2-3     | CNN                                | <https://blog.csdn.net/x2114754480/article/details/150401794> |
|             | 2-4     | PyTorch on CIFAR-10                | <https://blog.csdn.net/x2114754480/article/details/150459008> |
|             | 2-5     | Image Captioning with Vanilla RNNs | <https://blog.csdn.net/x2114754480/article/details/150938350> |
| Assignment3 | 3-1     | Image Captioning with Transformers | <https://blog.csdn.net/x2114754480/article/details/151654125> |
|             | 3-2     | Self-Supervised Learning           | <https://blog.csdn.net/x2114754480/article/details/151694699> |
|             | 3-3     | DDPM                               | <https://blog.csdn.net/x2114754480/article/details/151864627> |
|             | 3-4     | CLIP and DINO                      | <https://blog.csdn.net/x2114754480/article/details/151946399> |

## Getting started

If you need to directly obtain the code running results, you need to download the datasets and extract them to the corresponding path.

**CIFAR10 Dataset**

download from:

<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

extract to:

.\Assignments\assignment1\cs231n\datasets\cifar-10-batches-py

.\Assignments\assignment2\cs231n\datasets\cifar-10-batches-py

.\Assignments\assignment3\cs231n\datasets\cifar-10-batches-py

**Coco Dataset**

download from:

<https://cs231n.stanford.edu/coco_captioning.zip>

extract to:

.\Assignments\assignment2\cs231n\datasets\coco_captioning

.\Assignments\assignment3\cs231n\datasets\coco_captioning
