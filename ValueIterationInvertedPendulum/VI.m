%Value iteration
function [new_value, num_iterations] = VI(TOLERANCE, num_states, transition_probs, gamma, rewards, value)
    num_iterations = 0;
    new_value = zeros(num_states, 1);
    while true
        num_iterations += 1;
        %Asinchronous approach:
        for s = 1:num_states
            value1 = transition_probs(s, :, 1) * value;
            value2 = transition_probs(s, :, 2) * value;
            new_value(s) = max(value1, value2);
        end
        new_value = rewards + (gamma * new_value);
        %check convergence:
        diff = max(abs(new_value - value));
        if (diff < TOLERANCE)
            break;
        end
        value = new_value;
    end
endfunction