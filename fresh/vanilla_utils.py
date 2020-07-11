
def without(x, keys):
    if isinstance(x, dict):
        return {k:v for k, v in x.items()
                if k not in keys}

