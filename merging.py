def get_multiple_adapters_params(model, adapters):
    adapter_params = []
    for adapter in adapters:
        params = {
            name: param
            for name, param in model.lm.named_parameters()
            if adapter in name
        }
        adapter_params.append(params)
    return adapter_params


def get_adapter_params(model, adapter_name):
    return {
        name: param
        for name, param in model.lm.named_parameters()
        if adapter_name in name
    }


def merge_adapters(adapter_params, adapters, weights, merged_adapter):
    merged_params = {}
    # Assuming all adapters have the same keys

    keys = adapter_params[0].keys()
    first_adapter = adapters[0]

    for key in keys:
        val = 0
        for weight, adapter, params in zip(weights, adapters, adapter_params):
            new_key = key.replace(first_adapter, adapter)
            val += params[new_key] * weight
        merged_key = key.replace(first_adapter, merged_adapter)
        merged_params[merged_key] = val

    return merged_params


def overwrite_adapter_params(model, merged_params):
    for name, param in model.lm.named_parameters():
        if name in merged_params:
            param.data = merged_params[name]
