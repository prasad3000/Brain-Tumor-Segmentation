import torch.nn as nn 
import torch.nn.functional as F


class Autofocus(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches

        self.conv1 =nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=self.dilation_list[0])
        self.convatt11 = nn.Conv3d(inplanes1, int(inplanes1/2), kernel_size=kernel)
        self.convatt12 = nn.Conv3d(int(inplanes1/2), self.num_branches, kernel_size=1)
        self.bn_list1 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list1.append(nn.BatchNorm3d(outplanes1))
            
        self.conv2 =nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=self.dilation_list[0])
        self.convatt21 = nn.Conv3d(outplanes1, int(outplanes1/2), kernel_size=kernel)
        self.convatt22 = nn.Conv3d(int(outplanes1/2), self.num_branches, kernel_size=1)
        self.bn_list2 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))
        
        self.relu = nn.ReLU(inplace=True)
        if inplanes1==outplanes2:            
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:,:, 4:-4, 4:-4, 4:-4]
        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        att = self.relu(self.convatt11(feature))
        att = self.convatt12(att)
        att = F.softmax(att, dim=1)
        att = att[:,:,1:-1,1:-1,1:-1]

        # linear combination of different rates
        x1 = self.conv1(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1)* att[:,0:1,:,:,:].expand(shape)
        
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv1.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2* att[:,i:(i+1),:,:,:].expand(shape)
        
        x = self.relu(x1)
        
        # compute attention weights for the second autofocus layer
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        att2 = F.softmax(att2, dim=1)
        att2 = att2[:,:,1:-1,1:-1,1:-1]
        
        # linear combination of different rates
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21)* att2[:,0:1,:,:,:].expand(shape)
        
        for i in range(1, self.num_branches):
            x22 = F.conv3d(x, self.conv2.weight, padding =self.padding_list[i], dilation=self.dilation_list[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22* att2[:,i:(i+1),:,:,:].expand(shape)
                
        if self.downsample is not None:
            residual = self.downsample(residual)
     
        x = x21 + residual
        x = self.relu(x)
        return x
    
    
class AFN(nn.Module):
    def __init__(self, blocks, padding_list, dilation_list, channels, kernel_size , num_branches):
        super(AFN, self).__init__()
        
        # parameters in the architecture
        self.channels = channels     
        self.kernel_size = kernel_size
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.blocks = blocks
        
        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        
        self.layers = nn.ModuleList()
        for i in range(3):
            index = int(2 * i + 2)
            self.layers.append(Autofocus(self.channels[index], self.channels[index+1], self.channels[index+2], self.padding_list, self.dilation_list, self.num_branches))
           
        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        for layer in self.layers:
            x = layer(x)
             
        x = self.fc(x)
        return x
    
class MainLayer():
    def build_model(self, num_input = 4, num_classes = 5, num_branches = 4, padding_list = [0, 4, 8, 12], dilation_list = [2, 6, 10, 14]):
        channels = [num_input-1, 30, 30, 40, 40, 40, 40, 50, 50, num_classes]
        kernel = 3
                
        model = AFN(padding_list, dilation_list, channels, kernel, num_branches)
        return model