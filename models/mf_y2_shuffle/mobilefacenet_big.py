
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d,ReLU, Sequential, Module
from model import MobileFaceNet_sor, Linear_block, Conv_block, Depth_Wise, Residual
from models.facenet20.common_utility import L2Norm, Flatten


class ChannelShuffle(Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x

#correct shuffle version
class Depth_Wise_Shufflev2(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, shuffle_group = 3):
        super(Depth_Wise_Shufflev2, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups,  kernel=(1, 1), padding=(0, 0), stride=(1, 1), groups = shuffle_group )
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c,  kernel=(1, 1), padding=(0, 0), stride=(1, 1), groups = shuffle_group)
        self.residual = residual
        self.shuffle = ChannelShuffle(shuffle_group)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

#correct shuffle version
class ResidualShufflev2(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), shuffle_group = 3):
        super(ResidualShufflev2, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise_Shufflev2(c, c, residual=True, kernel=kernel, padding=padding,
                                                stride=stride, groups=groups, shuffle_group=shuffle_group))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)




class MobileFaceNet_y2(Module):
    def __init__(self, embedding_size, conv6_kernel = (7,7)):
        super(MobileFaceNet_y2, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual(64, num_block=2, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=8, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=4, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel= conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)

        out = self.l2(out)

        return out

#0.95 GMac 2.34 M
class MobileFaceNet_y2_shufflev2g4(MobileFaceNet_y2):
    def __init__(self, embedding_size=512, conv6_kernel=(7, 7)):
        super(MobileFaceNet_y2_shufflev2g4, self).__init__(embedding_size, conv6_kernel)
        val = 2
        shuffle_group = 4
        self.conv1 = Conv_block(3, int(64*val), kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ResidualShufflev2(int(64*val), num_block=1, groups=int(64*val), kernel=(3, 3), stride=(1, 1), padding=(1, 1), shuffle_group=shuffle_group)
        self.conv_23 = Depth_Wise(int(64*val), int(64*val), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(128*val))
        self.conv_3 = ResidualShufflev2(int(64*val), num_block=4, groups=int(128*val), kernel=(3, 3), stride=(1, 1), padding=(1, 1),shuffle_group=shuffle_group)
        self.conv_34 = Depth_Wise(int(64*val), int(128*val), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(256*val))
        self.conv_4 = ResidualShufflev2(int(128*val), num_block=6, groups=int(256*val), kernel=(3, 3), stride=(1, 1), padding=(1, 1), shuffle_group=shuffle_group)
        self.conv_45 = Depth_Wise(int(128*val), int(128*val), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(512*val))
        self.conv_5 = ResidualShufflev2(int(128*val), num_block=2, groups=int(256*val), kernel=(3, 3), stride=(1, 1), padding=(1, 1), shuffle_group=shuffle_group)
        self.conv_6_sep = Conv_block(int(128*val), int(512*val), kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(int(512*val), int(512*val), groups=int(512*val), kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(int(512*val), embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()
