
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
import numpy as np

class APA(nn.Module):
    def __init__(self, num_classes, prototype_dim):
        super(APA, self).__init__()
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.prototypes = nn.Parameter(torch.randn(128, prototype_dim))

    def forward(self, x, labels):
        # Compute the distances between the input features and the prototypes
        distances = torch.cdist(x, self.prototypes)
        
        # Compute the prototype assignments based on the distances
        assignments = torch.argmin(distances, dim=1)
        
        # Update the prototypes based on the assignments
        self.update_prototypes(x, assignments)
        
        return assignments

    def update_prototypes(self, x, assignments):
        # Perform prototype updating based on the input features and assignments
        for i in range(self.num_classes):
            assigned_samples = x[assignments == i]
            if len(assigned_samples) > 0:
                # self.prototypes[i] = torch.mean(assigned_samples, dim=0)
                mean_assigned_samples = torch.mean(assigned_samples, dim=0).cuda()
                prototypes_temp = self.prototypes.detach().clone().cuda()
                mean_assigned_samples = mean_assigned_samples.type(prototypes_temp.dtype)
                prototypes_temp.scatter_(0, torch.tensor([i]).cuda().unsqueeze(0), mean_assigned_samples.unsqueeze(0))
                self.prototypes = nn.Parameter(prototypes_temp).cuda()
class MF(nn.Module):
    def __init__(self, num_classes, memory_size):
        super(MF, self).__init__()
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, 128))
        self.linear_s = nn.Linear(128, 128)
        self.linear_f = nn.Linear(128, 1000)

    def forward(self, x, prototypes):
        # Compute the similarities between the input features and the memory
        # print(x.shape)
        x = self.linear_s(x)
        # print(x.shape)
        similarities = torch.matmul(x, self.memory.t())
        # print('similarities:',similarities.shape)
        # Compute the memory-based feedback
        # print('prototypes:',prototypes.shape)
        prototypes = self.linear_f(prototypes)
        # print('prototypes:',prototypes.shape)
        feedback = torch.matmul(similarities, prototypes.t())
        
        # Update the memory based on the learned prototypes
        self.update_memory(prototypes)
        
        return feedback

    def update_memory(self, prototypes):
        # Perform memory updating based on the learned prototypes
        # print('prototypes:',prototypes.shape)
        # print('self.memory:',self.memory.shape)
        # self.memory = torch.cat((self.memory, prototypes.t()), dim=0)
        self.memory = nn.Parameter(torch.cat((self.memory.detach(), prototypes.t()), dim=0))
        if self.memory.shape[0] > self.memory_size:
            self.memory = nn.Parameter(self.memory[-self.memory_size:])

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class SDSM(nn.Module):
    """
    selective discriminant space model (SDSM). Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(SDSM, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.simclr = nn.Linear(nf * 8 * block.expansion, 128)
        self.discriminant_projector = nn.Linear(512, 128)
        self.apa = APA(num_classes, 128)
        self.mf = MF(num_classes, 1000)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear
        self.classifier_0 = nn.Linear(128,512)

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f_train(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = avg_pool2d(out, out.shape[2])  
        out = out.view(out.size(0), -1)  
        return out

    def forward(self, x: torch.Tensor, labels,use_proj=False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = self.f_train(x)
        # print('out' ,out.shape)
        
        discriminant_features = self.discriminant_projector(out)
        # print('discriminant_features:',discriminant_features.shape)
        prototype_assignments = self.apa(discriminant_features, labels)
        # print('self.apa.prototypes:',self.apa.prototypes.shape)
        memory_feedback = self.mf(discriminant_features, self.apa.prototypes)
        # print('memory_feedback:',memory_feedback.shape)
        # print('memory_feedback:',memory_feedback.shape)
        # print('discriminant_features:',discriminant_features.shape)
        logits = self.classifier_0(discriminant_features + memory_feedback)
        # print('logits:',logits.shape)

        if use_proj:
            feature = logits
            out = self.simclr(logits)
            return feature, out
        else:
            out = self.linear(logits)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def SDSM18(nclasses: int, nf: int = 64) -> SDSM:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return SDSM(BasicBlock, [2, 2, 2, 2], nclasses, nf=64)


def init_weights(model, std=0.01):
    print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(model), std))
    for m in model.modules():
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, 0, std)
            if m.bias is not None:
                m.bias.data.zero_()
