%Main file:

%plotting parameters
time_steps_to_failure = [];
time = 0;
time_at_start_of_current_trial = 0;


num_states = 163;  %162 + failure state
failure_state = num_states;

num_failures = 0;

%%%% There are no observed transition probabilities or rewards.
%%%% These need to be learned from the model.

%for each state, account for prbability to get to next state 
%based on experience, depending on which of the two actions is chosen.
%Initially, assume a uniform distribution.
transition_probs = ones(num_states, num_states, 2) / num_states;
transition_counts = zeros(num_states, num_states, 2);

%similar for rewards:
reward_counts = zeros(num_states, 2);
rewards = zeros(num_states, 1);

%randomly initialze the value function array to some small values
value = rand(num_states, 1) * 0.1;

%set discount factor. The intuition is that to maximise the reward we would
%like to get positive rewards as soon as possible and postpone negative
%rewards for as long as possible.
gamma = 0.995;

%thresold for value iteration, ie check if it converged
TOLERANCE=0.01;

%maximum number of consecutive times that value iteration converged
%after just one iteration. If exceeded, the simulation will end and
%the algorithm is considered trained
max_no_learning = 20;

%initialze state:
x = 0.0;
x_dot = 0.0;
theta = 0.0;
theta_dot = 0.0;
state = what_state(x, x_dot, theta, theta_dot);

%animation parameters:
pause_time = 0.001;  %time between each frame; higher values will slow down the animation
%min_trial_length_to_start_display = 0;
%display_started=0;

animation(x, theta, pause_time);

consecutive_no_learning = 0;

%%start the algorithm: repeat until convergence
while consecutive_no_learning < max_no_learning

    %increment time_step
    time += 1;

    %1. compute policy based on learned transition probabilites
    pi_1 = transition_probs(state, :, 1) * value;
    pi_2 = transition_probs(state, :, 2) * value;
    if (pi_1 > pi_2)
        action = 1;
    elseif (pi_1 < pi_2)
        action = 2;
    else
        if rand < 0.5
            action = 1;
        else
            action = 2;
        end
    end
    %2. execute policy
    [x, x_dot, theta, theta_dot] = cart_pole(action, x, x_dot, theta, theta_dot);
    new_state = what_state(x, x_dot, theta, theta_dot);
    
    animation(x, theta, pause_time);
    
    %3. get rewards: -1 if the pole fell/cart out of bounds and 0 at every other state
    if (new_state == failure_state)
        R = -1;
    else
        R = 0;
    end
    %4. update the counters for transition probabilities and rewards
    transition_counts(state, new_state, action) += 1;
    reward_counts(new_state, 1) += R;
    reward_counts(new_state, 2) += 1;
    %5. In case of failure, you need to update the MDP (state transition probabilites
    % and rewards associated with each state).
    %In case these are zero, ie the state action pair has not been tried before,
    %maintain the uniform distribution.
    %After, you need to perform value iteration to get the optimal value function
    % for this MDP.
    if (new_state == failure_state)
        for a = 1:2
            for s = 1:num_states
                den = sum(transition_counts(s, :, a)); %times we took action a in state s
                if (den > 0)
                    transition_probs(s, :, a) = transition_counts(s, :, a) / den; %times we took action a in state s and got to s' /den
                end
                if (reward_counts(s, 2) > 0)
                    rewards(s) = reward_counts(s, 1) / reward_counts(s, 2);
                end
            end
        end
        %6. perfom value iteration.
        %keep account of the number of iterations of value iteration perfromed.
        %If it converged after just one iteration for a number of times, then there was
        %little learning left to do. If this happens max_no_learning times in a row
        %the algorithm should be finished training and the simulation can be stopped.
        [value, num_iterations] = VI(TOLERANCE, num_states, transition_probs, gamma, rewards, value);
        if (num_iterations == 1)
            consecutive_no_learning += 1;
        else
            consecutive_no_learning = 0;
        end
    end
    
    %update the state or re-initialize in case of failure
    %Also, in case simulation is taking too long, break out if faiulure > 500. Usually this shouldn't happen.
    %also, update plot parameters
    if (new_state == failure_state)
    
        num_failures += 1;
        time_steps_to_failure(num_failures) = time - time_at_start_of_current_trial;
        time_at_start_of_current_trial = time;
        
        %display some variables to keep track where in the simulation you are
        consecutive_no_learning
        num_failures
        time_steps_to_failure(num_failures)
        
        if (num_failures == 500)
            break;
        end
        x = -1.1 + rand(1)*2.2;
        x_dot = 0.0;
        theta = 0.0;
        theta_dot = 0.0;
        state = what_state(x, x_dot, theta, theta_dot);
    else 
        state=new_state;
    end
end

%learning curve
plot(log(time_steps_to_failure));
print -deps usingtheirmodel3.eps; %save the plot; current name is foo