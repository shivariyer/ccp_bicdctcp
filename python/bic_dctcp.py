# ****************************************************
# BIC+DCTCP algorithm, based on an online prediction
# of ECN and Capacity marker (CM).
#
# Author: Shiva R. Iyer
# Date: March 1, 2020
#
# ****************************************************

from __future__ import print_function

from sklearn import cluster
from sklearn import preprocessing

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
    REQ_BACKLOG_LEN = 100
    RANDOM_STATE = 1
    
    def __init__(self, datapath, datapath_info, cwnd_max):

        # cc parameters
        self.datapath = datapath
        self.datapath_info = datapath_info
        self.init_cwnd = int(self.datapath_info.mss * BIC_DCTCP_Flow.INIT_CWND)
        self.cwnd = self.init_cwnd
        self.cwnd_max = cwnd_max
        self.datapath.set_program("default", [("Cwnd", self.cwnd)])
        
        # model specific parameters
        # self.micros_prev = None
        self.weights = (0.5, 0.5)
        self.scaler = preprocessing.MinMaxScaler()
        self.data = []
        self.model = cluster.KMeans(2, random_state=BIC_DCTCP_Flow.RANDOM_STATE)
        
        log.debug('New flow, init_cwnd = {} bytes'.format(self.init_cwnd))
    
    def on_report(self, r):
        # add r.rtt to the list of recent N RTTs, use r.micros to
        # compute IAT and keep adding that also to the list
        # if micros_prev is None:
        #     micros_prev = r.micros
        # else:
        #     iat = r.micros - micros_prev
        #     micros_prev = r.micros
        
        log.debug('Received a report')
        rtt_feature = self.weights[0] * np.exp(r.rtt / 1e6)
        #iat_feature = self.weights[1] * np.exp(r.iat / 1e6)
        #self.data.append((rtt_feature, iat_feature))
        self.data.append(rtt_feature)
        if len(self.data) > BIC_DCTCP_Flow.REQ_BACKLOG_LEN:
            self.data.pop(0)
        
        if len(self.data) < BIC_DCTCP_Flow.REQ_BACKLOG_LEN:
        # if True:
            log.info('Following BIC+MD while list builds up')
            # TODO: fall back to AIMD or something in the meanwhile as the
            # list grows to required length
            if r.loss > 0 or r.sacked > 0:
            #if r.sacked > 0:
                log.debug('loss={}, sacked={}'.format(r.loss, r.sacked))
                #log.debug('sacked={}'.format(r.sacked))
                self.cwnd /= 2
                self.cwnd = max(self.cwnd, self.init_cwnd)
            else:
                self.cwnd += (self.cwnd_max - self.cwnd) / 2
                self.cwnd = min(self.cwnd, self.cwnd_max)
        else:
            # TODO: if the list of recent RTTs+IATs is pretty big, then do
            # a clustering and predict ECN for each packet (i.e. the
            # cluster group), and then compute fraction of ECN
            X = self.scaler.fit_transform(np.asarray(self.data))
            self.model.fit(X)
            ecn_bits = self.model.labels_.astype(bool)
            centroids = self.model.cluster_centers_
            if centroids[0] > centroids[1]:
                ecn_bits = ~ecn_bits
            
            log.debug('Cluster centers: {} {}'.format(centroids[0], centroids[1]))
            
            ecn_frac = ecn_bits.sum() / ecn_bits.size
            
            log.info('ecn_frac={}'.format(ecn_frac))
            
            # TODO: if the frac of ECN is < threshold, then do BIC
            # increase, else do DCTCP decrease
            if (ecn_frac == 0):
                if (r.loss > 0) or (r.sacked > 0):
                #if r.sacked == 0:
                    log.debug('loss={}, sacked={}'.format(r.loss, r.sacked))
                    #log.debug('sacked={}'.format(r.sacked))
                    self.cwnd /= 2
                    self.cwnd = max(self.cwnd, self.init_cwnd)
                else:
                    self.cwnd += (self.cwnd_max - self.cwnd) / 2
                    self.cwnd = min(self.cwnd, self.cwnd_max)
            else:
                log.debug('alpha={}'.format(ecn_frac))
                self.cwnd *= (1 - ecn_frac/2)
                self.cwnd = max(self.cwnd, self.init_cwnd)
        
        log.debug('New cwnd: {} bytes'.format(self.cwnd))
        self.datapath.update_field("Cwnd", int(self.cwnd))


class BIC_DCTCP(portus.AlgBase):
    
    def __init__(self, cwnd_max, agg_npkts=10):
        """ 
        cwnd_max: The upper limit on the congestion window. 
        
        version: The version of datapath program to use.
        
        '1' -- Report at every ack 
        
        '2' -- Report at specified aggregated interval
        
        agg_npkts: Aggregation resolution
        
        """
        super(BIC_DCTCP, self).__init__()
        self.cwnd_max = cwnd_max
        #self.version = version
        self.agg_npkts = agg_npkts
        log.debug('Created BIC_DCTCP class')

    def datapath_programs(self):
        return {"default" : """\
                            (def (Report
                                  (volatile rtt 0)
                                  (volatile iat 0)
                                  (volatile acked 0)
                                  (volatile sacked 0)
                                  (volatile loss 0)
                                  (volatile timeout false)
                                  ))
                            (when true
                              (:= Report.rtt Flow.rtt_sample_us)
                              (:= Report.iat Micros)
                              (:= Report.acked (+ Report.acked Ack.bytes_acked))
                              (:= Report.sacked (+ Report.sacked Ack.packets_misordered))
                              (:= Report.loss Ack.lost_pkts_sample)
                              (:= Report.timeout Flow.was_timeout)
                              (fallthrough)
                              )
                            (when (|| Report.timeout (> Report.loss 0))
                              (report)
                              (:= Micros 0)
                              )
                            (when (> Micros Flow.rtt_sample_us)
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
        return BIC_DCTCP_Flow(datapath, datapath_info, self.cwnd_max)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('BIC-DCTCP for mmWave flows')
    #parser.add_argument('version', type=int, choices=(1,2), help='\'1\' or \'2\'')
    parser.add_argument('cwnd_max', type=int, help='Max congestion window (bytes)')
    parser.add_argument('--ipc', choices=('netlink','unix'), default='netlink', help='Set type of ipc to use')
    parser.add_argument('--debug', action='store_true', help='Print debug messages')
    parser.add_argument('--log', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='DEBUG', help='Logging level (default: INFO)')
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
    
    fh = logging.FileHandler('logs/bic_dctcp_{}_{}.log'.format(args.cwnd_max, args.ipc))
    fh.setLevel(log_level_dict[args.log])
    
    formatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s [%(name)s] %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    
    #alg = BIC_DCTCP(args.cwnd_max, args.version)
    alg = BIC_DCTCP(args.cwnd_max)
    
    portus.start(args.ipc, alg, debug=args.debug)
