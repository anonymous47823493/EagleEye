#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 13 JULY 2020
#  ------------------------------------------------------------------
import torch
import distiller


def thinning(net, scheduler, input_tensor=None):
    scheduler.on_epoch_begin(1)
    scheduler.mask_all_weights()

    def create_graph(model):
        if input_tensor is not None:
            dummy_input = input_tensor
        else:
            dummy_input = torch.randn(16, 3, 32, 32)
        return distiller.SummaryGraph(model, dummy_input)

    sgraph = create_graph(net._net)
    from distiller.thinning import create_thinning_recipe_filters, apply_and_save_recipe

    thinning_recipe = create_thinning_recipe_filters(
        sgraph, net._net, scheduler.zeros_mask_dict
    )
    apply_and_save_recipe(
        net._net, scheduler.zeros_mask_dict, thinning_recipe, net.optimizer
    )
    net.optimizer.param_groups[0]['params'] = list(net._net.parameters())
    return net
