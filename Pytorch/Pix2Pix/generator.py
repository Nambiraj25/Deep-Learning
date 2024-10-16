import torch
from torch import nn
class UNETDown(nn.Module):
  def __init__(self,in_channels,out_channels,dropout=0.0,normalize=True):
    super(UNETDown,self).__init__()
    layers=[nn.Conv2d(in_channels,out_channels,4,2,1,bias=False)]
    if normalize:
      layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2,inplace=True))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model=nn.Sequential(*layers)
  def forward(self,x):
    return self.model(x)

class UNetUp(nn.Module):
  def __init__(self,in_channels,out_channels,dropout=0.0):
    super(UNetUp,self).__init__()
    layers=[
      nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    ] 
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model=nn.Sequential(*layers)
  def forward(self,x,skip_input):
    x=self.model(x)
    x=torch.cat((x,skip_input),1)
    return x
class Generator(nn.Module):
    def __init__(self,in_channels=3,out_channels=3):
      super(Generator,self).__init__()
      self.down1=UNETDown(in_channels,64,normalize=False)
      self.down2=UNETDown(64,128)
      self.down3=UNETDown(128,256)
      self.down4=UNETDown(256,512,dropout=0.5)
      self.down5=UNETDown(512,512,dropout=0.5)
      self.down6=UNETDown(512,512,dropout=0.5)
      self.down7=UNETDown(512,512,dropout=0.5)
      self.down8=UNETDown(512,512,normalize=False,dropout=0.5)

      self.up1=UNetUp(512,512,dropout=0.5)
      self.up2=UNetUp(1024,512,dropout=0.5)
      self.up3=UNetUp(1024,512,dropout=0.5)
      self.up4=UNetUp(1024,512,dropout=0.5)
      self.up5=UNetUp(1024,256)
      self.up6=UNetUp(512,128)
      self.up7=UNetUp(256,64)

      self.final=nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128,out_channels,3,1,1,bias=True),
        nn.Tanh()
      )
    def forward(self,x):
      d1=self.down1(x)
      d2=self.down2(d1)
      d3=self.down3(d2)
      d4=self.down4(d3)
      d5=self.down5(d4)
      d6=self.down6(d5)
      d7=self.down7(d6)
      d8=self.down8(d7)

      u1=self.up1(d8,d7)
      u2=self.up2(u1,d6)
      u3=self.up3(u2,d5)
      u4=self.up4(u3,d4)
      u5=self.up5(u4,d3)
      u6=self.up6(u5,d2)
      u7=self.up7(u6,d1)

      return self.final(u7)

def test_generator_shape():
    batch_size = 1
    in_channels = 3
    height = 512
    width = 512
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    generator = Generator(in_channels=in_channels, out_channels=3)

    output_tensor = generator(input_tensor)

    expected_shape = (batch_size, 3, height, width)
    output_shape = tuple(output_tensor.shape)  

    assert output_shape == expected_shape, f"Expected shape {expected_shape}, but got {output_shape}"
    print(f"Test passed: Output tensor shape is {output_shape}")
if __name__ == "__main__":
    test_generator_shape()