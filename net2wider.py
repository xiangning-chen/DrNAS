import torch
import torch.nn as nn
import torch.nn.functional as F


# bias = 0
def InChannelWider(module, new_channels, index=None):
    weight = module.weight
    in_channels = weight.size(1)

    if index is None:
        index = torch.randint(low=0, high=in_channels,
                              size=(new_channels-in_channels,))
    module.weight = nn.Parameter(
        torch.cat([weight, weight[:, index, :, :].clone()], dim=1), requires_grad=True)

    module.in_channels = new_channels
    module.weight.in_index = index
    module.weight.t = 'conv'
    if hasattr(weight, 'out_index'):
        module.weight.out_index = weight.out_index
    module.weight.raw_id = weight.raw_id if hasattr(
        weight, 'raw_id') else id(weight)
    return module, index


# bias = 0
def OutChannelWider(module, new_channels, index=None):
    weight = module.weight
    out_channels = weight.size(0)

    if index is None:
        index = torch.randint(low=0, high=out_channels,
                              size=(new_channels-out_channels,))
    module.weight = nn.Parameter(
        torch.cat([weight, weight[index, :, :, :].clone()], dim=0), requires_grad=True)

    module.out_channels = new_channels
    module.weight.out_index = index
    module.weight.t = 'conv'
    if hasattr(weight, 'in_index'):
        module.weight.in_index = weight.in_index
    module.weight.raw_id = weight.raw_id if hasattr(
        weight, 'raw_id') else id(weight)
    return module, index


def BNWider(module, new_features, index=None):
    running_mean = module.running_mean
    running_var = module.running_var
    if module.affine:
        weight = module.weight
        bias = module.bias
    num_features = module.num_features

    if index is None:
        index = torch.randint(low=0, high=num_features,
                              size=(new_features-num_features,))
    module.running_mean = torch.cat([running_mean, running_mean[index].clone()])
    module.running_var = torch.cat([running_var, running_var[index].clone()])
    if module.affine:
        module.weight = nn.Parameter(
            torch.cat([weight, weight[index].clone()], dim=0), requires_grad=True)
        module.bias = nn.Parameter(
            torch.cat([bias, bias[index].clone()], dim=0), requires_grad=True)
        
        module.weight.out_index = index
        module.bias.out_index = index
        module.weight.t = 'bn'
        module.bias.t = 'bn'
        module.weight.raw_id = weight.raw_id if hasattr(
            weight, 'raw_id') else id(weight)
        module.bias.raw_id = bias.raw_id if hasattr(
            bias, 'raw_id') else id(bias)
    module.num_features = new_features
    return module, index


def configure_optimizer(optimizer_old, optimizer_new):
    for i, p in enumerate(optimizer_new.param_groups[0]['params']):
        if not hasattr(p, 'raw_id'):
            optimizer_new.state[p] = optimizer_old.state[p]
            continue
        state_old = optimizer_old.state_dict()['state'][p.raw_id]
        state_new = optimizer_new.state[p]

        state_new['momentum_buffer'] = state_old['momentum_buffer']
        if p.t == 'bn':
            # BN layer
            state_new['momentum_buffer'] = torch.cat(
                [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id, p.out_index

        elif p.t == 'conv':
            # conv layer
            if hasattr(p, 'in_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][:, p.in_index, :, :].clone()], dim=1)
            if hasattr(p, 'out_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index, :, :, :].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id
            if hasattr(p, 'in_index'):
                del p.in_index
            if hasattr(p, 'out_index'):
                del p.out_index
    print('%d momemtum buffers loaded' % (i+1))
    return optimizer_new


def configure_scheduler(scheduler_old, scheduler_new):
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    print('scheduler loaded')
    return scheduler_new

   