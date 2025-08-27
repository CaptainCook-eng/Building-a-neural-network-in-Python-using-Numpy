
# formulas needed for Adam

def momentum_hat(forgetting_factor1, gradient, momentum):
    momentum = momentum
    momentum = forgetting_factor1 * momentum  + (forgetting_factor1 - 1) * gradient
    result = momentum / (1 - forgetting_factor1)
    return result

def second_moment_hat(forgetting_factor2, gradient, second_momentum):
    second_momentum = second_momentum
    second_momentum = forgetting_factor2 * second_momentum + (forgetting_factor2 - 1) * gradient**2
    result = second_momentum / (1 - forgetting_factor2)
    return result