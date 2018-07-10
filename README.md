# LinNet

Light Intuitive Neural Network - CNN C++练习

----

## 需补充

1. `layer.inl`中各层的`forward()`与`backward()`。注意`backward()`要利用到`forward`中已给出的`inputPtr`以利用输入数据计算参数的梯度。可以修改函数的参数列表以便于梯度在各层间的传播。

2. `optim.inl`中`OptimizerSGD`的实现，并完成在`Layer::backward()`与`OptimizerSGD`之间数据的传递。考虑如何利用`Optimizer::step()`执行Momentum梯度的计算。

3. 仿照`Net::train()`，完成`linnet.inl`中`Net::predict()`的实现。

4. 考虑用`cv::parallel_for_`及`cv::ParallelLoopBody`替换常见的`for`循环，以提高计算速度。
