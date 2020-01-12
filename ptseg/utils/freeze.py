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
    for name, param in model.classifier.named_parameters():
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
    print('model name not registered in freeze, no parameters will be trained.')
    return model
