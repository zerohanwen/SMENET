# from torchsummary import summary
# import pandas as  pd
# import numpy as np
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy
import math
from torch.nn import Parameter
# #from keras.utils import to_categorical
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score,f1_score
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
# from imblearn.over_sampling import SMOTE, ADASYN
from __main__ import device
def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride
        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


class AlexNet_ACmix(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.conv1dupy = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=320)
        self.conv1dupz = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=480)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1),
            nn.Softmax(),
            ACmix(4, 4, kernel_conv=3, stride=1),
        )
        self.ttt = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.Linear(15451,4096),
            # nn.ReLU(),
            # nn.Dropout(0.01),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Dropout(0.01),
            nn.Linear(25639, 4096),
            nn.Softmax(),
            nn.Linear(4096, 1024),
            nn.Softmax(),
            nn.Linear(1024, num_classes),
        )
        self.cat = nn.Sequential(
        )

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        z = self.conv1(y)
        z = self.relu(z)
        y = self.conv1dupy(y)
        y = self.relu(y)
        z = self.conv1dupz(z)
        z = self.relu(z)
        x = torch.cat((x, y, z), dim=1)
        x = x.reshape(x.shape[0], 1, 3, 1280)
        x = self.features(x)
        x = torch.flatten(x, 1)

        # x8 = self.features2(x4)
        # x16 = self.features3(x8)
        # x4 = torch.flatten(x1,1)
        # x8 = torch.flatten(x8,1)
        # x16 = torch.flatten(x16,1)
        # print(x4.shape)
        # print(x8.shape)
        # x = torch.cat((x4,x8,x16),dim=1)
        # print(x.shape)
        x = self.classifier(x)
        return x


class AlexNet_ACmix_1023(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.conv1dupy = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=320)
        self.conv1dupz = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=480)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1),

            nn.ReLU(),
            ACmix(4, 4, kernel_conv=3, stride=1),
        )

        self.ttt = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=3, padding=1),
        )

        self.classifier = nn.Sequential(
            # nn.MaxPool1d(kernel_size=3,stride=1),
            # nn.Linear(15451,4096),
            # nn.ReLU(),
            # nn.Dropout(0.01),
            # nn.Linear(256,512),
            # nn.ReLU(),
            # nn.Dropout(0.01),
            nn.Linear(34920, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(4096, num_classes),
        )
        self.cat = nn.Sequential(

        )

    def forward(self, x):
        t = x
        t = torch.flatten(t, 1)

        y = self.conv1(x)
        y = self.relu(y)
        y = self.max_pool(y)
        y = self.bn1(y)

        z = self.conv1(y)
        z = self.relu(z)
        z = self.max_pool(z)

        x = torch.cat((x, y, z), dim=2)
        x1 = x
        x2 = x
        x = torch.cat((x, x1, x2), dim=1)
        # x = x.reshape(2,1,3,1280)

        x = x.reshape(x.shape[0], 1, 3, 1680)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x, t), dim=1)

        # x8 = self.features2(x4)
        # x16 = self.features3(x8)
        # x4 = torch.flatten(x1,1)
        # x8 = torch.flatten(x8,1)
        # x16 = torch.flatten(x16,1)
        # print(x4.shape)
        # print(x8.shape)
        # x = torch.cat((x4,x8,x16),dim=1)
        # print(x.shape)
        x = self.classifier(x)

        return x


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.conv1dupy = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=320)
        self.conv1dupz = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=480)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1),

            nn.ReLU(),
            ACmix(4, 4, kernel_conv=3, stride=1),
        )

        self.ttt = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=3, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        self.cat = nn.Sequential(

        )

    def forward(self, x):

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


#########################
##########NET1025

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.001)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(0.001)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        #print(input_tensor)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Net_1025(nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)

        self.conv1dupy = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=320)
        self.conv1dupz = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=480)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1)
        self.selfatt = SelfAttention(4, 1280, 1280)
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        self.lstm = nn.LSTM(1280, hidden_size=640, bidirectional=True,batch_first=True)

    def forward(self, x):
        # t = x
        # t = torch.flatten(t, 1)
        #
        # y = self.conv1(x)
        # y = self.relu(y)
        # y = self.bn1(y)
        # y = self.conv1dupy(y)
        # y = self.relu(y)
        # y = self.bn1(y)
        #
        # z = self.conv1(y)
        # z = self.relu(z)
        # z = self.bn1(z)
        # z = self.conv1dupz(z)
        # z = self.relu(z)
        # z = self.bn1(z)
        #
        # x = torch.cat((x, y, z), dim=2)
        #print(x.shape)
        x, (hidden, cell) = self.lstm(x)
        #print(x.shape)
        x = self.selfatt(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #print(x.shape)
        return x




##########################
########UNET
class BiLSTM(nn.Module):
    def __init__(self,n_class,n_hidden):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        ##self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1*2, batch_size, 1280).to(device)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size, 1280).to(device)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        #print(outputs.shape)
        outputs = torch.reshape(outputs, (batch_size, 1, 2560))
        #model = self.fc(outputs)  # model : [batch_size, n_class]
        return outputs

class SoftWeighted(nn.Module):
    def __init__(self):
        super(SoftWeighted, self).__init__()
        self.num_view = 3
        self.weight_var = Parameter(torch.ones(3))

    def forward(self, data):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in range(self.num_view)]
        high_level_preds = 0
        for i in range(self.num_view):
            high_level_preds += weight_var[i] * data[i]

        return high_level_preds,weight_var

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class UNET(nn.Module):
    def __init__(self, num_classes):
        super(UNET, self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv1d(8, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()
        self.selfatt = SelfAttention(4, 2560, 2560)
        self.classifier = nn.Sequential(
            nn.Linear(2560, num_classes),
        )
        self.lstm = nn.LSTM(2560, hidden_size=1280, bidirectional=True, batch_first=True)
        self.BiLSTM = BiLSTM(n_class=2560, n_hidden=1280)
        self.soft = SoftWeighted()

    def forward(self, x):


        #print('1:',x.shape)

        conv1 = self.layer1_conv(x)
        #print('conv1:', conv1.shape)
        pool1 = F.max_pool1d(conv1, 2)
        #print('3:', pool1.shape)

        conv2 = self.layer2_conv(pool1)
        #print('4:', conv2.shape)
        pool2 = F.max_pool1d(conv2, 2)
        #print('5:', pool2.shape)

        conv3 = self.layer3_conv(pool2)
        #print('6:', conv3.shape)
        pool3 = F.max_pool1d(conv3, 2)
        #print('7:', pool3.shape)

        conv4 = self.layer4_conv(pool3)
        #print('8:', conv4.shape)
        pool4 = F.max_pool1d(conv4, 2)
        #print('9:', pool4.shape)

        conv5 = self.layer5_conv(pool4)
        #print('10:', conv5.shape)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.selfatt(outp)

        #y, (hidden, cell) = self.lstm(outp)

        y = self.BiLSTM(x)
        y = self.selfatt(y)

        z, (hidden, cell) = self.lstm(x)
        z = self.selfatt(z)

        x = self.selfatt(x)

        data = [x,outp,y]
        x,weight = self.soft(data)
        #print(weight)
        #x = torch.cat([x, outp, y], dim=2)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

########UNET
##################

##################
########bing_UNET

class bing_Unet(nn.Module):
    def __init__(self, num_classes):
        super(bing_Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv1d(8, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()
        self.selfatt = SelfAttention(4, 2560, 2560)
        self.classifier = nn.Sequential(
            # nn.Linear(2560, 512),
            # nn.ReLU(),
            # nn.Linear(512, num_classes)
            nn.Linear(2560, num_classes),
        )
        self.lstm = nn.LSTM(2560, hidden_size=1280, bidirectional=True, batch_first=True)

    def forward(self, x):


        #print('1:',x.shape)

        conv1 = self.layer1_conv(x)
        #print('conv1:', conv1.shape)
        pool1 = F.max_pool1d(conv1, 2)
        #print('3:', pool1.shape)

        conv2 = self.layer2_conv(pool1)
        #print('4:', conv2.shape)
        pool2 = F.max_pool1d(conv2, 2)
        #print('5:', pool2.shape)

        conv3 = self.layer3_conv(pool2)
        #print('6:', conv3.shape)
        pool3 = F.max_pool1d(conv3, 2)
        #print('7:', pool3.shape)

        conv4 = self.layer4_conv(pool3)
        #print('8:', conv4.shape)
        pool4 = F.max_pool1d(conv4, 2)
        #print('9:', pool4.shape)

        conv5 = self.layer5_conv(pool4)
        #print('10:', conv5.shape)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        #print(outp.shape) 2560
        outp = self.selfatt(outp)
        print(outp.shape)

        #y, (hidden, cell) = self.lstm(outp)
        # print(x.shape)
        #y = self.selfatt(y)
        x, (hidden, cell) = self.lstm(x)
        print(x.shape)
        x = self.selfatt(x)
        x = torch.cat([x, outp], dim=2)
        #x = torch.flatten(x, 1)

        x = self.classifier(x)
        print(x.shape)

        return x


##################
########bing_UNET
class SMENET_new(nn.Module):
    def __init__(self, num_classes):
        super(SMENET_new, self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv1d(8, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()
        self.selfatt = SelfAttention(4, 2560, 2560)
        #self.selfatt = SelfAttention(4, 1280, 1280)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
            #nn.Linear(1280, num_classes),
        )
        #self.lstm = nn.LSTM(2560, hidden_size=1280, bidirectional=True, batch_first=True)
        self.BiLSTM = BiLSTM(n_class=2560, n_hidden=1280) #2560
        #self.BiLSTM = BiLSTM(n_class=1280, n_hidden=640)  #1280
        self.soft = SoftWeighted()

    def forward(self, x):


        #print('1:',x.shape)

        conv1 = self.layer1_conv(x)
        #print('conv1:', conv1.shape)
        pool1 = F.max_pool1d(conv1, 2)
        #print('3:', pool1.shape)

        conv2 = self.layer2_conv(pool1)
        #print('4:', conv2.shape)
        pool2 = F.max_pool1d(conv2, 2)
        #print('5:', pool2.shape)

        conv3 = self.layer3_conv(pool2)
        #print('6:', conv3.shape)
        pool3 = F.max_pool1d(conv3, 2)
        #print('7:', pool3.shape)

        conv4 = self.layer4_conv(pool3)
        #print('8:', conv4.shape)
        pool4 = F.max_pool1d(conv4, 2)
        #print('9:', pool4.shape)

        conv5 = self.layer5_conv(pool4)
        #print('10:', conv5.shape)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        #outp = self.selfatt(outp)


        y = self.BiLSTM(x)
        #y = self.selfatt(y)

        # z, (hidden, cell) = self.lstm(x)
        # z = self.selfatt(z)

        # x = self.selfatt(x)


        data = [x,outp,y]
        x,weight = self.soft(data)


        #print(weight)
        # x = torch.cat([x, outp, y], dim=2)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #print(x)
        return x

########UNET
##################