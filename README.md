# A bandit algorithm for efficient on-farm research
by Yuji Saikai & Paul Mitchell

**Abstract**

Field experiment is the foundation for gaining agronomic insight into causal relationship between inputs and outputs and, based on them, optimizing cropping systems. When facilitated by precision agriculture, individual farmers may benefit from experimenting with new inputs and practices on their own heterogeneous fields. However, on-farm experiment is different from dedicated research activities in universities or corporations as it is often constrained by limited resources, requiring even higher efficiency, and needs to be incorporated into day-to-day operations, making a trade-off between exploration and exploitation. To address it, we develop an algorithm based on suitable machine learning techniques: active learning, multi-armed bandit, and Thompson sampling. Then, we demonstrate its advantage against the conventional practice using simulation built on the existing field trial data. Our algorithm uniformly outperforms it, and the efficiency gain could be substantial. In addition, the generality of our algorithm makes itself applicable to a wide range of similar problems.

[[pdf](mabps.pdf)]

![](fig1.png "Results")
