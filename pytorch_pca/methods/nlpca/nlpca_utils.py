import torch


def _conjugate_gradient_optimizer(net, train_out, max_iter, verbose):
    _, n_patterns = train_out.shape
    n_components = net.units_per_layer[0]
    eps = 1e-8

    weights = net.weights.clone().requires_grad_(True)

    prev_grad = None
    search_direction = None

    alpha_init = 0.01
    alpha_decay = 0.001

    for iteration in range(max_iter):
        scores = weights[: n_components * n_patterns].view(
            n_patterns, n_components
        )

        h = scores
        offset = n_components * n_patterns

        for i, (start, w_end, b_end) in enumerate(net.layer_indices):
            n_in = net.units_per_layer[i]
            n_out = net.units_per_layer[i + 1]

            w = weights[offset + start : offset + w_end].view(n_out, n_in)
            b = weights[offset + w_end : offset + b_end]

            h = torch.nn.functional.linear(h, w, b)
            h = torch.tanh(h)

        output = h.T

        error = (output - train_out) * net.data_dist
        mse = torch.sum(error**2) / (torch.sum(net.data_dist) + eps)

        weight_penalty = net.weight_decay * torch.sum(
            weights[n_components * n_patterns :] ** 2
        )
        total_loss = mse + weight_penalty

        grad = torch.autograd.grad(total_loss, weights, create_graph=False)[0]

        if iteration == 0:
            search_direction = -grad
        else:
            if prev_grad is not None and search_direction is not None:
                beta = torch.clamp(
                    torch.sum(grad**2) / (torch.sum(prev_grad**2) + eps),
                    0,
                    10,
                )
                search_direction = -grad + beta * search_direction
            else:
                search_direction = -grad

        alpha = alpha_init / (1 + iteration * alpha_decay)

        with torch.no_grad():
            weights = weights + alpha * search_direction
            weights.requires_grad_(True)

        prev_grad = grad.detach()

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}")

    return weights


def _compute_r2_vectorized(data, scores, net, scaling_factor):
    n_components = scores.shape[1]
    valid_data = torch.nan_to_num(data, 0.0)
    data_mask = ~torch.isnan(data)
    TSS = torch.sum(valid_data**2 * data_mask.float())
    R2cum = torch.zeros(n_components)

    weights_list = []
    biases_list = []

    if net.weights is None:
        raise ValueError("Network weights are not initialized")

    for start, w_end, b_end in net.layer_indices:
        n_in = net.units_per_layer[len(weights_list)]
        n_out = net.units_per_layer[len(weights_list) + 1]

        w = net.weights[start:w_end].view(n_out, n_in)
        b = net.weights[w_end:b_end]
        weights_list.append(w)
        biases_list.append(b)

    for i in range(1, n_components + 1):
        scores_subset = scores[:, :i]
        h = scores_subset

        w_first = weights_list[0][:, :i]
        h = torch.nn.functional.linear(h, w_first, biases_list[0])
        h = torch.tanh(h)

        for j in range(1, len(weights_list)):
            h = torch.nn.functional.linear(h, weights_list[j], biases_list[j])
            h = torch.tanh(h)

        reconstructed = h / scaling_factor
        diff = valid_data - reconstructed
        RSS = torch.sum((diff**2) * data_mask.float())

        if TSS > 1e-10:
            R2cum[i - 1] = torch.clamp(1 - RSS / TSS, 0.0, 1.0)
        else:
            R2cum[i - 1] = 0.0

    return R2cum
