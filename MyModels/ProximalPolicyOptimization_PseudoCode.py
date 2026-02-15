# Initialization: Set the pre-trained LLM parameters as θ_old, and copy them to θ
θ_old = PretrainedLLM.parameters
θ = copy(θ_old)

# Sampling Phase: Generate a batch of data using θ_old
for each prompt in dataset:
    trajectory = []
    state = prompt
    while not end_of_sequence:
        token, logpi_old = θ_old.generate_token(state)
        # Record the current state, token, and the log-probability under θ_old
        trajectory.append( (state, token, logpi_old, reward, V(state)) )
        state = state + token  # Update the state (append token)
    store trajectory

# Compute Advantages (e.g., using GAE)
for each trajectory:
    for t from last token downto first:
        δ_t = reward[t] + γ * V(state[t+1]) - V(state[t])
        A_t = δ_t + γ * λ * A[t+1]  # Recursively compute the advantage

# PPO Update Phase: Multiple epochs
for each PPO update epoch:
    for each token data (state s_t, token a_t, logpi_old, A_t) in batch:
        # 1. Compute the log-probability under the current policy
        logpi_current = θ.log_probability(s_t, a_t)
        # 2. Calculate the probability ratio
        r_t = exp( logpi_current - logpi_old )
        # 3. Compute the unclipped and clipped objectives
        loss_unclipped = r_t * A_t
        loss_clipped = clip(r_t, 1-ε, 1+ε) * A_t
        # 4. The token loss is the negative of the minimum of these two values
        loss_token = -min(loss_unclipped, loss_clipped)
    # 5. Average the loss over all tokens and perform a gradient update
    θ = Update(θ, average(loss_token over batch))

# After updating, copy θ to θ_old for the next round of sampling
θ_old = copy(θ)