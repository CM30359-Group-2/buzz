import replay

def fill_expert_buffer():
    # For each demo, parse the demo and add transitions to the replay buffer
    pass

def parse_demo():
    pass

def add_transitions():
    pass

def define_action_dict():
    pass

def make_env():
    pass

def build_model():
    pass

def train_model():
    pass

if __name__ == "main":
    # Create the environment
    env, n_actions = make_env(
        game="CarRacing-v2",
        stack=True,
        scale_rew=False,
        scenario="contest",
        is_content_env=False,
        action_dict=define_action_dict(),
    )

    # Fill expert buffer
    expert_buffer = replay.PrioritizedReplayBuffer(500000)
    expert_buffer = fill_expert_buffer(expert_buffer)

    # Build models
    pre_train_model = build_model(n_actions)
    target_model = build_model(n_actions)

    # Pre-train model
    target_model, expert_buffer = train_model(pre_train_model, target_model, expert_buffer, n_actions, batch_size=32, train_steps=750000),

    # Copy model to target model

    # Train model

