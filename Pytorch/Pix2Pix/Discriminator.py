import torch
from torch import nn
class Discriminator(nn.Module):
  def __init__(self,in_channels=3):
    super(Discriminator,self).__init__()
    def discriminator_block(in_channels,out_channels,normalize=True):
      layers=[nn.Conv2d(in_channels,out_channels,4,2,1,bias=False)]
      if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.LeakyReLU(0.2,inplace=True))
      return layers  
    self.model=nn.Sequential(
      *discriminator_block(in_channels*2,64,normalize=False),
      *discriminator_block(64,128),
      *discriminator_block(128,256),
      *discriminator_block(256,512),
      nn.ZeroPad2d((1,0,1,0)),
      nn.Conv2d(512,1,4,1,1,bias=False)
    )
  def forward(self,img,target):
    img_input=torch.cat((img,target),1)
    return self.model(img_input)
def test_discriminator_shape():
  batch_size=1
  in_channels=3
  height=256
  width=256

  img=torch.randn(batch_size,in_channels,height,width)
  target=torch.randn(batch_size,in_channels,height,width)
   
  discriminator=Discriminator(in_channels=in_channels)

  output=discriminator(img,target)

  expected_shape=(batch_size,1,16,16)

  output_shape=tuple(output.shape)
  assert output_shape==expected_shape,f"Expected shape {expected_shape},but got {output_shape}"
  print(f"Test passed: Output tensor shape is {output_shape}")
if __name__=="__main__":
  test_discriminator_shape()