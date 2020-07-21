# BIC-DCTCP congestion control algorithm

This is the code for the implementation of a modified BIC-DCTCP
congestion control algorithm over
[CCP](https://ccp-project.github.io/). BIC stands for _Binary
InCrease_, which means that the congestion window (_cwnd_) increases
by 50% of the difference between the current window and the max
window. [DCTCP](https://people.csail.mit.edu/alizadeh/papers/dctcp-sigcomm10.pdf)
is a well-known algorithm, very similar to default TCP, which utilizes
an Explicit Congestion Notification (ECN) bit marked on each packet to
control the sending rate. In our implementation, we assume that the
ECN bits are not marked, and so we run an online unsupervised
algorithm at the sender to predict impending congestion and the ECN
bits. The fraction of packets with predicted ECN 1 is computed over
the last few packets, and this fraction, denoted as $\alpha$, is used
to control the _cwnd_ decrease in the algorithm.

# Requirements

Rust (nightly version), CCP datapath integration, Portus, Python 2,
NumPy.

Follow the instructions in the CCP guide
[here](https://ccp-project.github.io/ccp-guide/setup/index.html) under
"Setup" to install both Rust, the CCP kernel module and Portus.

**Note that Python 3 is not supported at this time.**


# Running the algorithm

- Load the CCP kernel module as step 2
  [here](https://ccp-project.github.io/ccp-guide/setup/index.html).
  
- Start our user space algorithm as `sudo python bic_dctcp.py`, as
  shown
  [here](https://ccp-project.github.io/ccp-guide/running.html). Mandatory
  argument to the algorithm is the _cwnd max_ in bytes, necessary for
  the BIC algorithm. The _backlog_ argument specifies length of
  history (past packet RTTs) for computing the ECN fraction.


# Status

We only have Python implementation of our algorithm currently. The
Rust implementation is incomplete and under development.
