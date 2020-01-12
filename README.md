# Exp Logs

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

with bias decay

transforms resize384, crop320, jitter0.4

weight decay 5e-4

**test prec 76.5**

![](MarkdownPic/0112-1010.png)


## 0112-11-25

changes compared with 0112-10-10

no bias decay
```python
# no bias decay
param_optimizer = list(filter(lambda p: p.requires_grad, model.parameters()))
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.001},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
]
optimizer = torch.optim.SGD(optimizer_grouped_parameters,
                            lr=args.lr, momentum=0.9, weight_decay=5e-4)

# original optimizer
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                             lr=args.lr, momentum=0.9, weight_decay=5e-4)
```

**test prec 76.95**

![](MarkdownPic/0112-1125.png)