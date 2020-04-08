extern crate portus;
extern crate rusty_machine as rm;
extern crate slog;

use portus::ipc::Ipc;
use portus::lang::Scope;
use portus::{CongAlg, Datapath, DatapathInfo, DatapathTrait, Report};
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;
use rusty_machine::linalg::Matrix;
use slog::debug;
use std::collections::HashMap;

// #[derive(Clone)]
// pub enum Constant {
//     Cwnd(u32),
//     Rate { rate: u32, cwnd_cap: u32 },
// }

pub struct CcpBicDctcpAlg {
    pub logger: Option<slog::Logger>,
    pub cwnd_max: u32,
    pub init_cwnd: u32,
}

pub struct CcpBicDctcpFlow {
    logger: Option<slog::Logger>,
    sc: Scope,
}

impl<I: Ipc> CongAlg<I> for CcpBicDctcpAlg {
    type Flow = CcpBicDctcpFlow;

    fn name() -> &'static str {
        "bicdctcp"
    }

    fn datapath_programs(&self) -> HashMap<&'static str, String> {
        let mut h = HashMap::default();
        h.insert(
            "bicdctcp",
            "(def (Report
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
               (:= Report.sacked Ack.packets_misordered))
               (:= Report.loss Ack.lost_pkts_sample)
               (:= Report.timeout Flow.was_timeout)
               (report)
               (:= Micros 0)
               )"
            .to_owned(),
        );

        h
    }

    fn new_flow(&self, mut control: Datapath<I>, info: DatapathInfo) -> Self::Flow {
        let init_cwnd = if self.init_cwnd != 0 {
            self.init_cwnd
        } else {
            info.init_cwnd
        };
        let params = vec![("Cwnd", init_cwnd)];
        let sc = control.set_program("bicdctcp", Some(&params)).unwrap();
        CcpBicDctcpFlow {
            logger: self.logger.clone(),
            sc,
        }
    }
}

impl portus::Flow for CcpBicDctcpFlow {
    fn on_report(&mut self, _sock_id: u32, m: Report) {
        let rtt = m
            .get_field("Report.rtt", &self.sc)
            .expect("expected rtt in report") as u32;
        let iat = m
            .get_field("Report.iat", &self.sc)
            .expect("expected iat in report") as u32;
        let acked = m
            .get_field("Report.acked", &self.sc)
            .expect("expected acked in report") as u32;
        let sacked = m
            .get_field("Report.sacked", &self.sc)
            .expect("expected sacked in report") as u32;
        let loss = m
            .get_field("Report.loss", &self.sc)
            .expect("expected loss in report") as u32;

        self.logger.as_ref().map(|log| {
            debug!(log, "report";
                   "rtt(us)" => rtt,
                   "iat(us)" => iat,
                   "acked(bytes)" => acked,
                   "sacked(pkts)" => sacked,
                   "loss(pkts)" => loss,
            );
        });
    }
}
