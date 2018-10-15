import torch
from torch.autograd import Variable as V
#输入输出 y = 10x+1
x_data = V(torch.randn(100,1)+10*torch.ones(100,1))
y_data = x_data*10+1
print(x_data)
print(y_data)
# x_data =V(torch.Tensor([[1.0],[2.0],[3.0],[4.0],[5.0]]))
# y_data =V(torch.Tensor([[5.0],[7.0],[9.0],[11.0],[13.0]]))
#利用pytorch建立一个神经网络
class Linear1(torch.nn.Module):
    def __init__(self):
        super(Linear1, self).__init__()
        self.linear = torch.nn.Linear(1,1) #INPUT AND OUTPUT
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
#定义一个网络
torch.optim
model = Linear1()
criterion = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0)
torch.optim.Adam
#开始批训练数据
for epoch in range(1000):
    #利用优化器（optima）进行反向传播训练
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
# 测试框架
tes = V(torch.Tensor([[0.5215]]))
print(model(tes))
print(model.linear.weight.data)
print(model.linear.bias.data)
