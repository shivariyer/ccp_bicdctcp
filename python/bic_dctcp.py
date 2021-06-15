# ****************************************************
# BIC+DCTCP algorithm, based on an online prediction
# of an ECN-like marker.
#
# Author: Shiva R. Iyer
# Date: March 1, 2020
#
# ****************************************************

from __future__ import print_function

import portus
import numpy as np

import logging
import coloredlogs
import argparse


class BIC_DCTCP_Flow():
    INIT_CWND = 10
    BACKLOG_LEN_MAX = 100
    RANDOM_STATE = 1
    
    def __init__(self, datapath, datapath_info, cwnd_max, alpha_thres, backlog, preprocess):
        
        # cc parameters
        self.datapath = datapath
        self.datapath_info = datapath_info
        self.init_cwnd = float(self.datapath_info.mss * BIC_DCTCP_Flow.INIT_CWND)
        self.cwnd = self.init_cwnd
        self.cwnd_max = cwnd_max
        self.alpha_thres = alpha_thres
        assert 0 <= alpha_thres < 1

        self.backlog = backlog
        if self.backlog > BIC_DCTCP_Flow.BACKLOG_LEN_MAX:
            self.backlog = BIC_DCTCP_Flow.BACKLOG_LEN_MAX

        if preprocess is None:
            self.preprocess = lambda x: x
        else:
            if preprocess == 'exp':
                self.preprocess = np.exp
            elif preprocess == 'tanh':
                self.preprocess = np.tanh

        self.datapath.set_program("default", [("Cwnd", self.cwnd)])
        
        # model specific parameters
        self.weights = (0.5, 0.5)
        self.data = np.empty(self.backlog) * np.nan
        self.centroids = ()
        self.counter = 0
        self.rtt_max = 0
        self.rtt_min = np.inf
        self.alpha = 0
        self.max_centroid_dist = 0
        
        log.info('New flow, init_cwnd = {} bytes'.format(self.init_cwnd))
    
    def on_report(self, r):
        if r.loss > 0 or r.sacked > 0:
            #self.cwnd /= 2
            self.cwnd *= (1 - self.alpha/2)
            self.cwnd = max(self.cwnd, self.init_cwnd)
            log.info('PACKET LOSS / MISORDER, loss: {}, sacked: {}, alpha: {}, cwnd: {} bytes'
                     .format(r.loss, r.sacked, self.alpha, self.cwnd))
        else:
            if r.rtt < self.rtt_min:
                self.rtt_min = r.rtt
            if r.rtt > self.rtt_max:
                self.rtt_max = r.rtt

            # rtt is in microseconds (us)
            #rtt_feature = r.rtt / 1e6
            #rtt_feature = np.exp(r.rtt / 1e6)
            #rtt_feature = np.tanh(r.rtt / 1e6)
            rtt_feature = self.preprocess(r.rtt / 1e6)
            self.data[self.counter] = rtt_feature
            self.counter = (self.counter+1) % self.backlog
            rtt_vecn = 0
            if len(self.centroids) == 0:
                self.centroids = (rtt_feature,)
            elif len(self.centroids) == 1:
                # ensuring that first two centroids are always
                # distinct, AND that centroid_0 is always less than
                # centroid_1
                if rtt_feature > self.centroids[0]:
                    self.centroids = (self.centroids[0], rtt_feature)
                elif rtt_feature < self.centroids[0]:
                    self.centroids = (rtt_feature, self.centroids[0])
            else:
                # which cluster does current RTT data point belong to?
                # (if it is closer to centroid_0, the smaller one,
                # then vecn is 0, else vecn is 1)
                rtt_vecn = int((abs(rtt_feature - self.centroids[0]) > abs(rtt_feature - self.centroids[1])))

                # select only valid data in the backlog
                data_valid = self.data[np.isfinite(self.data)]

                # compute distances to old centroid and assign labels
                dist_0 = np.fabs(data_valid - self.centroids[0])
                dist_1 = np.fabs(data_valid - self.centroids[1])
                labels = (dist_0 > dist_1) # cluster 1 if dist_0 > dist_1, cluster 0 else

                # compute the new centroids after the new assignment
                centroid_0 = data_valid[~labels].mean() if (~labels).any() else self.centroids[0]
                centroid_1 = data_valid[labels].mean() if labels.any() else self.centroids[1]

                # if existing dist between centroids is greater, then
                # don't update the cluster centroids, else update the
                # centroids
                if self.max_centroid_dist == 0:
                    self.max_centroid_dist = abs(centroid_0 - centroid_1)
                elif self.max_centroid_dist < abs(centroid_0 - centroid_1):
                    self.max_centroid_dist = abs(centroid_0 - centroid_1)
                    self.centroids = (centroid_0, centroid_1)

                # cluster 0 is NOCONG, cluster 1 is CONG
                n_vecn_NOCONG, n_vecn_CONG = (~labels).sum(), labels.sum()

                # if that assumption is wrong, then swap the cluster counts
                # if self.centroids[0] > self.centroids[1]:
                #     n_vecn_NOCONG, n_vecn_CONG = n_vecn_CONG, n_vecn_NOCONG

                # compute the alpha
                self.alpha = n_vecn_CONG/float(labels.size)
                
                log.debug('n_vecn_0: {}, n_vecn_1: {}, alpha: {}'.format(n_vecn_NOCONG, n_vecn_CONG, self.alpha))
                log.debug('Centroids: ({:.7f}, {:.7f})'.format(*self.centroids))

                action = None
            if self.alpha > self.alpha_thres:
                self.cwnd *= (1 - self.alpha/2)
                self.cwnd = max(self.cwnd, self.init_cwnd)
                action = 'DECREASE'
            else:
                self.cwnd += (self.cwnd_max - self.cwnd) / 2
                self.cwnd = min(self.cwnd, self.cwnd_max)
                action = 'INCREASE'
            log.info('{}, cur_rtt: {} us, cur_rtt_vecn: {}, rtt_feature: {:.7f}, max_rtt: {} us, min_rtt: {} us, alpha: {}, cwnd: {} bytes'
                     .format(action, r.rtt, rtt_vecn, rtt_feature, self.rtt_max, self.rtt_min, self.alpha, self.cwnd))
        self.datapath.update_field("Cwnd", int(self.cwnd))


class BIC_DCTCP(portus.AlgBase):
    
    def __init__(self, cwnd_max, alpha_thres, backlog, preprocess):
        """ 
        cwnd_max: The upper limit on the congestion window. 

        alpha_thres: Threshold on fraction of packets marked with VECN=1 before cwnd is reduced

        backlog: Length of the backlog (i.e. amount of RTT history maintained)

        preprocess: Preprocessing function to apply to RTT feature
        
        """
        super(BIC_DCTCP, self).__init__()
        self.cwnd_max = cwnd_max
        self.alpha_thres = alpha_thres
        self.backlog = backlog
        self.preprocess = preprocess
        log.debug('Created BIC_DCTCP class')

    def datapath_programs(self):
        return {"default" : """\
                            (def (Report
                                  (rtt 0)
                                  (iat 0)
                                  (volatile acked 0)
                                  (volatile sacked 0)
                                  (volatile loss 0)
                                  (volatile timeout false)
                                  ))
                            (when true
                              (:= Report.rtt Flow.rtt_sample_us)
                              (:= Report.iat Micros)
                              (:= Report.acked Ack.bytes_acked)
                              (:= Report.sacked Ack.packets_misordered)
                              (:= Report.loss Ack.lost_pkts_sample)
                              (:= Report.timeout Flow.was_timeout)
                              (report)
                              (:= Micros 0)
                            )
            """
            }
    
    def new_flow(self, datapath, datapath_info):
        return BIC_DCTCP_Flow(datapath, datapath_info, self.cwnd_max, self.alpha_thres, self.backlog, self.preprocess)


def frac_type(arg):
    val = float(arg)
    if not 0 < val < 1:
        raise argparse.ArgumentTypeError('expected a fraction arg in (0,1)')
    return val


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('BIC-DCTCP for mmWave flows')
    parser.add_argument('cwnd_max', type=int, help='Max congestion window (bytes)')
    parser.add_argument('--alpha-thres', '-A', type=frac_type, default=0.5, help='Threshold on alpha before cwnd is reduced')
    parser.add_argument('--backlog', '-B', type=int, default=10, help='Length of backlog for clustering')
    parser.add_argument('--preprocess', '-P', choices=('tanh', 'exp'), help='Optional preprocessing for RTT features')
    parser.add_argument('--ipc', choices=('netlink','unix'), default='netlink', help='Set type of ipc to use')
    parser.add_argument('--debug', action='store_true', help='Print debug messages')
    parser.add_argument('--log', '-L', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file-path', '-F', dest='logfilepath',
                        help='File in which to log console output')
    args = parser.parse_args()
    
    log_level_dict = {'DEBUG' : logging.DEBUG,
                      'INFO' : logging.INFO,
                      'WARNING' : logging.WARNING,
                      'ERROR' : logging.ERROR,
                      'CRITICAL' : logging.CRITICAL}
    
    log = logging.getLogger('BIC+DCTCP')
    log.setLevel(log_level_dict[args.log])
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level_dict[args.log])

    if args.logfilepath is None:
        args.logfilepath = 'logs/bicdctcp_{}_a{:02.0f}_b{:03d}_P{}_{}.log'.format(args.cwnd_max, args.alpha_thres*10, args.backlog, args.preprocess, args.ipc)
    print('Logging console output to "{}"'.format(args.logfilepath))
    fh = logging.FileHandler(args.logfilepath)
    fh.setLevel(log_level_dict[args.log])

    colorFormatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s %(created)f [%(name)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    simpleFormatter = logging.Formatter(fmt='%(asctime)s %(created)f [%(name)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
        
    ch.setFormatter(colorFormatter)
    fh.setFormatter(simpleFormatter)
    log.addHandler(ch)
    log.addHandler(fh)
    
    alg = BIC_DCTCP(args.cwnd_max, args.alpha_thres, args.backlog, args.preprocess)
    
    portus.start(args.ipc, alg, debug=args.debug)
