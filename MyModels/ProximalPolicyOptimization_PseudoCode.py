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
        # cf. 지금은 reward를 trajectory에 포함시키고 있지만, 실제로는 reward 계산이 rollout과 별도의 과정으로 진행될 수 있음. 전체 시퀀스(문장) 완성 후에 시퀀스 단위로 reward를 부여하고, 추후 토큰 단위로 reward 계산을 별도로 진행하는 경우도 있음.
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
        # 가장 초기학습에서는 θ와 θ_old가 거의 동일하므로 logpi_current와 logpi_old는 유사할 것이다. 즉, policy rate = 1에 가까울 것.
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