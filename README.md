# Policy Gradients
Policy Gradient is the technique of directly using function approximators to generate a policy. These policies are stochastic meaning they give a probability distribution over all available actions for a given state. 
$$\pi_{\theta}(s,a) = P[a | s. \theta]$$

WE use the average reward per time step as objective function that we want to optimize for improving the policy.

$$J_{avgR}(\theta) = E[\sum_{t=0}^{T-1}r_{t+1}|\pi_{\theta}]$$

$$=\sum_{t=i}^{T-1}P(s_t,a_t| \tau)r_{t+1}$$

Where i is the arbitrary starting point in the trajectory and $P(s_t,a_t|\tau)$ is the probability if taking $a_t$ after reaching state $s_t$ following policy $\pi_{\theta}$ in trajectory $\tau$. This probability can be expanded as.
$$J_{avgR}(\theta) = \sum_{s}d_{\pi_{\theta}}(s)\sum_{a}\pi_{\theta}(a|s)R_{s}^{a}$$

$d_{\pi_{\theta}}(s)$ is the probability of reaching state s if following policy $\pi$. if $J(\theta)$ is objective function. Policy gradient optimizes it using gradient of this objective function

$$\nabla \theta = \alpha \nabla_{\theta}J(\theta)$$

$$\nabla_{\theta}J(\theta) = (\frac{\partial J(\theta)}{\theta_1},...,\frac{\partial J(\theta)}{\theta_n})^T$$

$$\nabla_{\theta} \pi_{\theta}(a|s) = \pi_{\theta}(a|s) \frac{\nabla_{\theta}\pi_{\theta}(a|s)}{\pi_{\theta}(a|s)}$$

$$\nabla_{\theta} \pi_{\theta}(a|s) =  \pi_{\theta}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s)$$

This simply says that we are increasing the likely hood of previously taken actions in our policy. We will make this increase proportional to the reward received form that action, That means increase the likely hood of those previously taken actions proportional to the rewards received after taking that action.
\\ Now we know 

$$J(\theta) = \sum_{s \in S}d_{\pi_{\theta}}(s)\sum_{a \in A}\pi_{\theta}(a|s)R_{s}^{a}$$

$$\nabla_{\theta}J(\theta) = \sum_{s \in S}d_{\pi_{\theta}}(s)\sum_{a \in A}\pi_{\theta}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s)R_{s}^{a}$$

This can be written as an expectation
$$= E_{\pi_{\theta}}[\nabla_{\theta} \log \pi_{\theta}(a|s)R]$$

Here R is the actual monte carlo return. To write this more generally.

$$J(\theta) = E_{\pi_{\theta}}[\nabla_{\theta} \log \pi_{\theta}(A_t|S_t)Q_{\pi_\theta}(S_t,A_t)]$$

We can have a seperate regression problems for Q function $Q_{\phi}(S_t,A_t)$ parametrized by $\phi$ which acts as a critic which provides less variance in the returns 
and a baseline to evaluate the policy against.

Since $J(\theta)$ is a performance measure rather than a penalty/loss measure like other machine learning cost functions, we seek to maximize it and therefore we use stochastic gradient ascent. In the code for reinforce we therefore use the negative of this objective function and use stochastic gradient descent.
