# Machine Learning Library

Special thanks to @jeremiah-corrado for being my mentor this summer, answering Chapel questions, and helping with implementation correctness.

During my time at HPE, I was tasked with creating a machine learning library from scratch using Python, then translating it to Chapel. 

The `python` directory contains the Python implementation of the library, and the `chapel` directory contains the analogous Chapel implementation. A more up-to-date version of the Chapel implementation can be found in the Chapel lang repository [here](https://github.com/chapel-lang/chapel) in the directory `test/studies/ml/lib`. 

These programs aren't exactly the same, but they are very similar. The Chapel implementation is more complete, and the Python implementation is more for reference and testing.



# Usage

### Python
```bash
$ cd python
$ python3 trainer.py {dataset_size}
# or
$ python3 mnist_trainer.py {dataset_size}
```

### Chapel
This would compile with the Chapel 1.32 version of the compiler ([at commit 60c2e02](https://github.com/chapel-lang/chapel/tree/60c2e02d2667c584f84356d748c66b3ae0daf000)).  
```bash
$ cd chapel
$ chpl --fast trainer.chpl
$ ./trainer --dataSize={dataset_size}
# or
$ chpl --fast mnist_trainer.chpl
$ ./mnist_trainer --dataSize={dataset_size}
```

# Resources

- [CNN From Scratch With NumPy](https://www.kaggle.com/code/milan400/cnn-from-scratch-numpy)
- [Dive Into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)
- [CNN Backpropagation Notes](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
- [Neural Networks from Scrath](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Backpropagation Simplified](https://towardsdatascience.com/back-propagation-simplified-218430e21ad0)
- [A Survey on the New Generation of Deep Learning in Image Processing](https://www.researchgate.net/publication/337746202_A_Survey_on_the_New_Generation_of_Deep_Learning_in_Image_Processing)
