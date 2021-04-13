import torch 
import torch.nn as nn




class singleconv(nn.Module):
    def __init__(self,inc,outc,bn=True,dp=True,**kwargs):
        super(singleconv,self).__init__()
        self.conv = nn.Conv2d(in_channels=inc,out_channels=outc,bias=False,**kwargs)
        self.bn = bn
        self.dp = dp
        self.batchn = nn.BatchNorm2d(outc)
        self.drop = nn.Dropout2d(0.2,inplace=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)


    def forward(self,x):
        x = self.conv(x)
        if self.bn:
            x = self.batchn(x)
        if self.dp:
            x = self.drop(x)
        
        return(self.relu(x))


class iniconv(nn.Module):
    def __init__(self,inc,outc,**kwargs):
        super(iniconv,self).__init__()
        self.conv1 = singleconv(inc,outc,bn=False,dp=False,**kwargs)
        self.conv2 = singleconv(outc,outc,bn=True,dp=False,**kwargs)
        self.out = singleconv(outc*2,outc,bn=True,dp=True,**kwargs)

    def forward(self,x,y):
        x = self.conv1(x)
        x = self.conv2(x)

        y = self.conv1(y)
        y = self.conv2(y)

        z = torch.cat([x,y],dim=1)

        return self.out(z)





class T_enhancer(nn.Module):
    def __init__(self,inchannels=3,f=16):
        super(T_enhancer,self).__init__()

        self.first = iniconv(inchannels,inchannels,kernel_size=3,stride=1,padding=1)

        self.block11 = singleconv(inchannels,f,bn=True,dp=False,kernel_size=3,stride=1,padding=1)
        self.block12 = singleconv(f,f*2,bn=True,dp=False,kernel_size=3,stride=1,padding=1)
        self.block13 = singleconv(f*2,f*4,bn=True,dp=True,kernel_size=3,stride=1,padding=1)

        self.block14 = singleconv(inchannels,f*2,bn=True,dp=False,kernel_size=5,stride=1,padding=2)
        self.block15 = singleconv(f*2,f*4,bn=True,dp=False,kernel_size=5,stride=1,padding=2)
        self.block16 = singleconv(f*4,f*4,bn=True,dp=True,kernel_size=5,stride=1,padding=2)


        self.block21 = singleconv(f*4,f*4,bn=True,dp=True,kernel_size=3,stride=1,padding=1)
        self.block22 = singleconv(f*4,f*2,bn=True,dp=True,kernel_size=3,stride=1,padding=1)
        self.block23 = singleconv(f*2,f,bn=True,dp=False,kernel_size=3,stride=1,padding=1)

        self.block24 = singleconv(f*4,f*2,bn=True,dp=True,kernel_size=5,stride=1,padding=2)
        self.block25 = singleconv(f*2,f,bn=True,dp=False,kernel_size=5,stride=1,padding=2)

        self.block26 = singleconv(f*4,f,bn=True,dp=False,kernel_size=7,stride=1,padding=3)

        self.block27 = singleconv(f*4,f,bn=True,dp=False,kernel_size=3,stride=1,padding=1)

        self.block31 = singleconv(f,f,bn=True,dp=False,kernel_size=3,stride=1,padding=1)

        self.block41 = singleconv(f,f//2,bn=True,dp=True,kernel_size=3,stride=1,padding=1)
        self.block42 = singleconv(f//2,f//2,bn=True,dp=False,kernel_size=3,stride=1,padding=1)
        self.block43 = singleconv(f//2,inchannels,bn=True,dp=False,kernel_size=3,stride=1,padding=1)
        self.last = nn.Tanh()




    def forward(self,x,y):
        inp = x

        x = self.first(x,y)
        x1 = self.block11(x)
        x1 = self.block12(x1)
        x1 = self.block13(x1)


        x2 = self.block14(x)
        x2_1 = self.block15(x2)
        x2 = self.block16(x2_1)

        x = torch.add(x1,x2_1)
        x = torch.add(x,x2)

        x1 = self.block21(x)
        x1 = self.block22(x1)
        x1 = self.block23(x1)


        x2 = self.block24(x)
        x2 = self.block25(x2)

        x3 = self.block26(x)

        x4 = self.block27(x)

        x = torch.add(x1,x2)
        x = torch.add(x,x3)

        x1 = self.block31(x)

        x = torch.add(x,x1)
        x =torch.add(x,x4)

        x = self.block41(x)
        x = self.block42(x)
        x = self.block43(x)
        x = self.last(x)

        # r1,r2,r3 = torch.split(x,3,dim=1)
        # inp = inp+r1*(torch.pow(inp,2)-inp)
        # inp = inp+r2*(torch.pow(inp,2)-inp)
        # inp = inp+r3*(torch.pow(inp,2)-inp)

        return x




def testt():
    inp1 = torch.randn((1,3,512,512)).cuda()
    inp2 = torch.randn((1,3,512,512)).cuda()
    model = T_enhancer().cuda()
    out = model(inp1,inp2)

    print(inp1.shape,out.shape)

    input_names = ['l_rbg','l_hsv']
    output_names = ['E_rgb']
    torch.onnx.export(model,(inp1,inp2),'/home/tejas/experimentations/image_enhancement/T_enhancer.onnx', input_names=input_names, output_names=output_names)


# testt()


class SingleConv(nn.Module):
    def __init__(self,inchannels,outchannels,bn=True,dropout=True,**kwargs):
        super(SingleConv,self).__init__()
        self.conv1 =nn.Conv2d(in_channels=inchannels,out_channels=outchannels,bias= not bn,**kwargs)
        self.bn = nn.BatchNorm2d(outchannels)
        self.dropout = nn.Dropout2d(0.2)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.b = bn
        self.d = dropout


    def forward(self,x):
        x = self.conv1(x)
        if self.b:
            x = self.bn(x)
        if self.d:
            x = self.dropout(x) 
        x = self.relu(x)
        return x 


class Discriminator(nn.Module):
    def __init__(self,inchannels=6,features=[64,128,256,512]):
        super(Discriminator,self).__init__()
        self.downs = nn.ModuleList()

        #down
        for feature in features:
            dstate,bstate=False if feature == features[-1] else True,False if feature == features[0] else True
            s = 1 if feature == features[-1] else 2
            self.downs.append(SingleConv(inchannels=inchannels,outchannels=feature,dropout=dstate,bn=bstate,kernel_size=4,stride = s,padding=1,padding_mode='reflect'))
            inchannels=feature 
        self.downs.append(nn.Conv2d(features[-1],1,kernel_size=3,stride=1,padding=1,padding_mode='reflect'))

    def forward(self,x,y):
        x = torch.cat((x,y),dim=1)

        for down in self.downs:
            x = down(x)

        return x


def testd():
    # with torch.cuda.amp.autocast():
    inp1 = torch.randn((1,3,512,512)).cuda()
    inp2 = torch.randn((1,3,512,512)).cuda()
    model = Discriminator(features=[64,128,256,512]).cuda()
    out = model(inp1,inp2)
    input_names = ['x','y']
    output_names = ['yhat']
    torch.onnx.export(model,(inp1,inp2),'/home/tejas/experimentations/image_enhancement/disc.onnx', input_names=input_names, output_names=output_names)

    print(inp1.shape,inp2.shape,out.shape)



# testd()

