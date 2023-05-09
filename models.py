import torch
from torch import nn
import torch.nn.functional as F


factors=[1,1,1,1,1/2,1/4,1/8,1/16,1/32]

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Dense_block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, normalization=True):
        super(Dense_block, self).__init__()
        self.normalization = normalization
        self.dense = nn.Linear(input_dim, output_dim)
        if (self.normalization):
            self.norm = nn.BatchNorm1d(output_dim)
        self.activate = activation_func(activation)

    def forward(self, x):
        x = self.dense(x)
        if (self.normalization):
            x = self.norm(x)
        x = self.activate(x)

        return x


class WSConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,gain=2):
        super().__init__()
        self.conv= nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.scale= (gain/(in_channels*kernel_size**2))**0.5
        self.bias=self.conv.bias
        self.conv.bias=None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.conv(x * self.scale) +self.bias.view(1,self.bias.shape[0],1,1)


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon=1e-8

    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_pixelnorm=True):
        super().__init__()
        self.conv1=WSConv2d(in_channels,out_channels)
        self.conv2=WSConv2d(out_channels,out_channels)
        self.leaky=nn.LeakyReLU(0.2)
        self.pn=PixelNorm()
        self.use_pn=use_pixelnorm

    def forward(self,x):
        x=self.leaky(self.conv1(x))
        x=self.pn(x) if self.use_pn else x
        x=self.leaky(self.conv2(x))
        x=self.pn(x) if self.use_pn else x

        return x

class Generator(nn.Module):
    def __init__(self,z_dim, n_classes, in_channels,img_channels=3):
        super().__init__()
        #The beginning of the model is different from the rest, so that we define the initial layer

        #added for ACGAN
        self.label_emb = nn.Embedding(n_classes, z_dim)

        self.initial=nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim*2,in_channels,4,1,0), #1x1 to 4x4
            #We mutiply by 2 the z_dim because we concat the noise vector with the embedding
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb=WSConv2d(in_channels,img_channels,kernel_size=1,stride=1, padding=0)
        self.prog_blocks,self.rgb_layers=nn.ModuleList(),nn.ModuleList([self.initial_rgb])

        for i in range(len(factors)-1):
            # factors[i]--> factors[i+1]
            conv_in_c=int(in_channels*factors[i])
            conv_out_c=int(in_channels*factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c,img_channels,kernel_size=1,stride=1,padding=0))

    def fade_in(self,alpha,upscaled,generated):
        return torch.tanh(alpha*generated+(1-alpha)*upscaled)

    def forward(self,x,label,alpha,steps):
        label_embedding=self.label_emb(label)
        label_embedding=label_embedding.unsqueeze(2).unsqueeze(3)
        x=torch.cat([x,label_embedding],dim=1)
        out=self.initial(x) #4x4

        if steps==0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled=F.interpolate(out,scale_factor=2,mode="nearest")
            out=self.prog_blocks[step](upscaled)

        final_upscaled=self.rgb_layers[steps-1](upscaled)
        final_out=self.rgb_layers[steps](out)

        return self.fade_in(alpha,final_upscaled,final_out)



class Discriminator(nn.Module):
    def __init__(self,in_channels,n_classes,img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers, self.embeddings = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.leaky=nn.LeakyReLU(0.2)

        resolutions=[4,8,16,32,64,128,256,512,1024]

        for resolution in resolutions:
            self.embeddings.append(nn.Embedding(n_classes,resolution*resolution))

        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c,use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels+1, conv_in_c, kernel_size=1, stride=1,padding=0))


        self.initial_rgb=WSConv2d(img_channels+1,in_channels,kernel_size=1,stride=1,padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool=nn.AvgPool2d(kernel_size=2,stride=2)


        self.final_block=nn.Sequential(
            WSConv2d(in_channels+1,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,kernel_size=4,padding=0,stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,1,kernel_size=1,padding=0,stride=1)
        )

    def fade_in(self,alpha,downscaled,out):
        return alpha*out+(1-alpha)*downscaled

    def minibatch_std(self,x):
        batch_statistics=torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1)

    def forward(self,x,labels,alpha,steps):
        cur_step=len(self.prog_blocks) - steps
        embeddings=self.embeddings[steps](labels)
        embeddings=embeddings.view(labels.shape[0],1,x.shape[2],x.shape[2]) #x.shape[2] is the image size [Batch_size,channels,imgsize,imgsize]
        x=torch.cat([x,embeddings],dim=1)
        out=self.leaky(self.rgb_layers[cur_step](x))

        if steps==0:
            out=self.minibatch_std(out)
            out = self.final_block(out).view(out.shape[0], -1)
            return out

        downscaled= self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out=self.avg_pool(self.prog_blocks[cur_step](out))
        out=self.fade_in(alpha,downscaled,out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out=self.prog_blocks[step](out)
            out=self.avg_pool(out)

        out= self.minibatch_std(out)

        out=self.final_block(out).view(out.shape[0],-1)
        return out

