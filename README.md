#Exp Logs

## TODO List

- [ ] try other net, vgg, inception, resnext
- [ ] focal loss
- [ ] label smoothing

## 0112-10-10

**to be the baseline of senet152**

epoch 25

lr 0.05

decay 10 16 22

arch senet152, freeze and free layer4, fc

bias decay

transforms resize384, crop320, jitter0.4

**test prec 76.5**

![](MarkdownPic/0112-1010.png)
