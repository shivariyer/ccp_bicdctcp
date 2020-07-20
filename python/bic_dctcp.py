# ****************************************************
# BIC+DCTCP algorithm, based on an online prediction
# of ECN and Capacity marker (CM).
#
# Author: Shiva R. Iyer
# Date: March 1, 2020
#
# ****************************************************

from __future__ import print_function

#from sklearn import cluster
#from sklearn import preprocessing

import portus
import numpy as np

import logging
import coloredlogs
import argparse

class Dataset(object):

    def __init__(self, X):
        self.X_orig = np.asarray(X)
        self.X = self.X_orig
    
    def rescale(self):
        scaler = preprocessing.MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        return self
    
    def weight(self, ws):
        ws = np.asarray(ws)
        if (ws.ndim > 1) or (ws.size != self.X.shape[1]):
            raise Exception('Invalid weights!')
        self.X = (self.X * ws) / ws.sum()
        return self
        
    def transform(self, func):
        self.X = func(self.X)
        return self


class BIC_DCTCP_Flow():
    INIT_CWND = 10
    BACKLOG_LEN_MAX = 100
    RANDOM_STATE = 1
    
    def __init__(self, datapath, datapath_info, cwnd_max, backlog):
        
        # cc parameters
        self.datapath = datapath
        self.datapath_info = datapath_info
        self.init_cwnd = float(self.datapath_info.mss * BIC_DCTCP_Flow.INIT_CWND)
        self.cwnd = self.init_cwnd
        self.cwnd_max = cwnd_max

        self.backlog = backlog
        if self.backlog > BIC_DCTCP_Flow.BACKLOG_LEN_MAX:
            self.backlog = BIC_DCTCP_Flow.BACKLOG_LEN_MAX

        self.datapath.set_program("default", [("Cwnd", self.cwnd)])
        
        # model specific parameters
        # self.micros_prev = None
        self.weights = (0.5, 0.5)
        #self.scaler = preprocessing.MinMaxScaler()
        self.data = np.empty(self.backlog) * np.nan
        #self.model = cluster.KMeans(2, random_state=BIC_DCTCP_Flow.RANDOM_STATE)
        self.centroids = ()
        self.counter = 0
        self.rtt_max = 0
        self.rtt_min = np.inf
        self.ecn_frac = 0
        self.max_centroid_dist = 0
        
        log.info('New flow, init_cwnd = {} bytes'.format(self.init_cwnd))
    
    def on_report(self, r):
        if r.loss > 0 or r.sacked > 0:
            #self.cwnd /= 2
            self.cwnd *= (1 - self.ecn_frac/2)
            self.cwnd = max(self.cwnd, self.init_cwnd)
            log.info("packet loss, loss={}, sacked={}, ecn_frac={}, cwnd={}".format(r.loss, r.sacked, self.ecn_frac, self.cwnd))
        else:
            if r.rtt < self.rtt_min:
                self.rtt_min = r.rtt
            if r.rtt > self.rtt_max:
                self.rtt_max = r.rtt
            # rtt is in microseconds (us)
            #rtt_feature = np.exp(r.rtt / 1e6)
            rtt_feature = np.tanh(r.rtt / 1e6)
            self.data[self.counter] = rtt_feature
            self.counter = (self.counter+1) % self.backlog
            if len(self.centroids) == 0:
                self.centroids = (rtt_feature,)
            elif len(self.centroids) == 1:
                self.centroids = (self.centroids[0], rtt_feature)
            else:
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

                # assume cluster 0 is CONG, cluster 1 is NOCONG
                n_ecn_NOCONG, n_ecn_CONG = labels.sum(), (~labels).sum()

                # if that assumption is wrong, then swap the cluster counts
                if self.centroids[0] < self.centroids[1]:
                    n_ecn_NOCONG, n_ecn_CONG = n_ecn_CONG, n_ecn_NOCONG

                # compute the ECN frac
                self.ecn_frac = n_ecn_CONG/float(labels.size)
                
                log.debug('n_ecn_0: {}, n_ecn_1: {}, ecn_frac: {}'.format(n_ecn_NOCONG, n_ecn_CONG, self.ecn_frac))
                log.debug('Centroids: ({:.7f}, {:.7f})'.format(*self.centroids))
            if self.ecn_frac > 1:
                self.cwnd *= (1 - self.ecn_frac/2)
                self.cwnd = max(self.cwnd, self.init_cwnd)
            else:
                self.cwnd += (self.cwnd_max - self.cwnd) / 2
                self.cwnd = min(self.cwnd, self.cwnd_max)
            log.info('cur rtt: {} us, rtt_feature: {:.7f}, max rtt: {} us, min rtt: {} us, ecn_frac: {}, cwnd: {} bytes'
                      .format(r.rtt, rtt_feature, self.rtt_max, self.rtt_min, self.ecn_frac, self.cwnd))
        self.datapath.update_field("Cwnd", int(self.cwnd))


class BIC_DCTCP(portus.AlgBase):
    
    def __init__(self, cwnd_max, backlog, agg_npkts=10):
        """ 
        cwnd_max: The upper limit on the congestion window. 
        
        version: The version of datapath program to use.
        
        '1' -- Report at every ack 
        
        '2' -- Report at specified aggregated interval
        
        agg_npkts: Aggregation resolution
        
        """
        super(BIC_DCTCP, self).__init__()
        self.cwnd_max = cwnd_max
        self.backlog = backlog
        #self.version = version
        self.agg_npkts = agg_npkts
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
    
    # def list_datapath_programs(self):
    #     # TODO: ask Akshay if Flow.rtt_sample_us is an average or
    #     # is the RTT of the most recent Ack
    #     log.info('Using version {}'.format(self.version))
        
    #     if self.version == 1:
    #         return {"default" : """\
    #                         (def (Report
    #                               (rtt 0)
    #                               (iat 0)
    #                               (volatile acked 0)
    #                               (volatile sacked 0)
    #                               (volatile loss 0)
    #                               (volatile timeout false)
    #                               ))
    #                         (when true
    #                           (:= Report.rtt Flow.rtt_sample_us)
    #                           (:= Report.iat Micros)
    #                           (:= Report.acked Ack.bytes_acked)
    #                           (:= Report.sacked Ack.packets_misordered))
    #                           (:= Report.loss Ack.lost_pkts_sample)
    #                           (:= Report.timeout Flow.was_timeout)
    #                           (report)
    #                           (:= Micros 0)
    #                           )
    #         """
    #         }
    #     elif self.version == 2:
    #         return {"default" : """\
    #                         (def (Report
    #                               (rtt 0)
    #                               (micros_prev 0)
    #                               (micros_cur 0)
    #                               (iat 0)
    #                               (volatile acked 0)
    #                               (volatile sacked 0)
    #                               (volatile loss 0)
    #                               (volatile timeout false)
    #                               )
    #                              (count 0)
    #                              (volatile iat_cur 0)
    #                              )
    #                         (when true
    #                           (:= count (+ count 1))
    #                           (:= Report.rtt Flow.rtt_sample_us)
    #                           (:= Report.micros_cur Micros)
    #                           (if (> Report.micros_prev 0)
    #                             (:= iat_cur (- micros_cur micros_prev))
    #                             (if (> count 0)
    #                               (:= Report.iat (/ (+ iat_cur (* Report.iat (- count 1))) count)
    #                               )
    #                             )
    #                           (if (== Report.micros_prev 0)
    #                             (:= Report.micros_prev Report.micros_cur)
    #                             )
    #                           (:= Report.acked Ack.bytes_acked)
    #                           (:= Report.sacked Ack.packets_misordered))
    #                           (:= Report.loss Ack.lost_pkts_sample)
    #                           (:= Report.timeout Flow.was_timeout)
    #                           (fallthrough)
    #                           )
    #                         (when (> Ack.packets_acked {0})
    #                           (report)
    #                           (:= Micros 0)
    #                           )
    #                         (when (|| Report.timeout (> Report.loss 0))
    #                           (report)
    #                           (:= Micros 0)
    #                           )
    #         """.format(agg_npkts-1)
    #         }
    #     else:
    #         raise Exception('BIC_DCTCP: Unsupported version')
    
    def new_flow(self, datapath, datapath_info):
        return BIC_DCTCP_Flow(datapath, datapath_info, self.cwnd_max, self.backlog)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('BIC-DCTCP for mmWave flows')
    #parser.add_argument('version', type=int, choices=(1,2), help='\'1\' or \'2\'')
    parser.add_argument('cwnd_max', type=int, help='Max congestion window (bytes)')
    parser.add_argument('--backlog', '-k', default=10, help='Length of backlog for clustering')
    parser.add_argument('--ipc', choices=('netlink','unix'), default='netlink', help='Set type of ipc to use')
    parser.add_argument('--debug', action='store_true', help='Print debug messages')
    parser.add_argument('--log', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='DEBUG', help='Logging level (default: INFO)')
    parser.add_argument('--log-file-path', dest='logfilepath',
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
        args.logfilepath = 'logs/bic_dctcp_{}_{}.log'.format(args.cwnd_max, args.ipc)
    print('Logging console output to "{}"'.format(args.logfilepath))
    fh = logging.FileHandler(args.logfilepath)
    fh.setLevel(log_level_dict[args.log])

    colorFormatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s [%(name)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    simpleFormatter = logging.Formatter(fmt='%(asctime)s [%(name)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    
    ch.setFormatter(colorFormatter)
    fh.setFormatter(simpleFormatter)
    log.addHandler(ch)
    log.addHandler(fh)
    
    #alg = BIC_DCTCP(args.cwnd_max, args.version)
    alg = BIC_DCTCP(args.cwnd_max, args.backlog)
    
    portus.start(args.ipc, alg, debug=args.debug)
