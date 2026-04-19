# Code Snippets

### Find Best Layer

class CNN(torch.nn.Module):

&#x20;   def \_\_init\_\_(self,num\_layers):

&#x20;       super().\_\_init\_\_()

&#x20;       layers = \[]

&#x20;       in\_channels = 3

&#x20;       out\_channels = 16

&#x20;       for i in range(num\_layers):

&#x20;           layers.append(torch.nn.Conv2d(in\_channels=in\_channels, out\_channels=out\_channels,

&#x20;                           kernel\_size=3, padding='same'))

&#x20;           layers.append(torch.nn.ReLU())

&#x20;           layers.append(torch.nn.MaxPool2d(kernel\_size=2))

&#x20;           in\_channels = out\_channels

&#x20;           if i+1 < num\_layers:

&#x20;               out\_channels += 16

&#x20;       layers.append(torch.nn.Flatten())

&#x20;       layers.append(torch.nn.LazyLinear(256))

&#x20;       layers.append(torch.nn.ReLU())

&#x20;       layers.append(torch.nn.Linear(256, 10))



&#x20;       self.model = torch.nn.Sequential(

&#x20;           \*layers,

&#x20;       )

&#x20;   def forward(self,x):

&#x20;       return self.model(x)



### Find Best Feature Maps



