def calculate_total_patches(config):
    """
    Calculates the exact number of patches per image based on the policy.
    """
    H, W = config.image_shape[:2]
    ph, pw = config.patch_size
    
    if config.transmission_policy_name == "Laplacian":
        # Sum patches for all levels
        levels = config.transmission_kwargs.get('pyramid_levels', [2, 1, 0])
        total = 0
        for lvl in levels:
            scale = 2 ** lvl
            # Use Ceil Division for grid dimensions
            gh = (H // scale + ph - 1) // ph
            gw = (W // scale + pw - 1) // pw
            total += gh * gw
        return total
    
    else:
        # Raw / Zlib (Single Level)
        gh = (H + ph - 1) // ph
        gw = (W + pw - 1) // pw
        return gh * gw