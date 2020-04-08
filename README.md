# BIC-DCTCP congestion control algorithm

This is the code for the implementation of a modified BIC-DCTCP congestion control algorithm over [CCP](https://ccp-project.github.io/). BIC stands for _Binary InCrease_, which means that the congestion window increases by 50% of the difference between the current window and the max window. [DCTCP](https://people.csail.mit.edu/alizadeh/papers/dctcp-sigcomm10.pdf) is a well-known protocol which utilizes an Explicit Congestion Notification (ECN) bit marked on each packet to control the sending rate. In our implementation, we assume that the ECN bits are not marked, and so we regularly run an online prediction at the sender to predict impending congestion and the ECN bits, which we then use in our protocol. 

This project is currently under development.
