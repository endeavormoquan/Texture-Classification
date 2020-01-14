import torch


def free_resnet34(model):
    for name, param in model.layer4[1].named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[2].named_parameters():
        param.requires_grad = True
    for name, param in model.avgpool.named_parameters():
        param.requires_grad = True
    for name, param in model.fc.named_parameters():
        param.requires_grad = True
    return model


def free_resnet50(model):
    for name, param in model.layer4[0].named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[1].named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[2].named_parameters():
        param.requires_grad = True
    for name, param in model.avgpool.named_parameters():
        param.requires_grad = True
    for name, param in model.fc.named_parameters():
        param.requires_grad = True
    return model


def free_vgg(model):
    """
    (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    )
    """
    # for name, param in model.classifier.named_parameters():
    #     param.requires_grad = True
    for name, param in model.classifier[3].named_parameters():
        param.requires_grad = True
    for name, param in model.classifier[6].named_parameters():
        param.requires_grad = True
    return model


def free_senet152(model):
    # for name, param in model.layer4[2].named_parameters():
    #     param.requires_grad = True
    for name, param in model.fc.named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[2].named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[1].named_parameters():
        param.requires_grad = True
    for name, param in model.layer4[0].named_parameters():
        param.requires_grad = True
    return model


def free_inception(model):
    for name, param in model.fc.named_parameters():
        param.requires_grad = True
    for name, param in model.Mixed_7c.named_parameters():
        param.requires_grad = True
    return model


def free_nts(model):
    for name, param in model.concat_net.named_parameters():
        param.requires_grad = True
    for name, param in model.partcls_net.named_parameters():
        param.requires_grad = True
    return model


def freeze_params(model_name, model):
    for param in model.parameters():
        param.requires_grad = False
    if model_name == 'resnet34':
        model = free_resnet34(model)
        return model
    if model_name == 'resnet50':
        model = free_resnet50(model)
        return model
    if model_name == 'resnet101':
        # layers in block4 in resnet50, resnet101 and resnet152 is the same
        model = free_resnet50(model)
        return model
    if model_name == 'resnet152':
        model = free_resnet50(model)
        return model
    if model_name == 'vgg19bn':
        model = free_vgg(model)
        return model
    if model_name == 'vgg16bn':
        model = free_vgg(model)
        return model
    if model_name == 'senet152':
        model = free_senet152(model)
        return model
    if model_name == 'se_resnet50':
        model = free_senet152(model)
        return model
    if model_name == 'inception':
        model = free_inception(model)
        return model
    if model_name == 'nts':
        model = free_nts(model)
        return model
    print('model name not registered in freeze, no parameters will be trained.')
    return model
