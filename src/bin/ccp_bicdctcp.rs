extern crate ccp_bicdctcp;
extern crate clap;
extern crate portus;
extern crate slog;

use ccp_bicdctcp::CcpBicDctcpAlg;
use clap::Arg;
use slog::warn;

const MBPS_TO_BPS: u32 = 1_000_000;
const BITS_TO_BYTES: u32 = 8;
const PKTS_TO_BYTES: u32 = 1500;
fn main() {
    let log = portus::algs::make_logger();

    let (cfg, ipc) = || -> Result<(CcpBicDctcpAlg, String), String> {
        let matches = clap::App::new("CCP BIC+DCTCP")
            .version("0.1.0")
            .author("Shiva R. Iyer <shiva.iyer@cs.nyu.edu>")
            .about("Congestion control algorithm which uses a combination of BIC and DCTCP")
            .arg(
                Arg::with_name("ipc")
                    .long("ipc")
                    .takes_value(true)
                    .required(true)
                    .help("Sets the type of ipc to use: (netlink|unix)")
                    .validator(portus::algs::ipc_valid),
            )
            .arg(
                Arg::with_name("cwnd_max")
                    .long("cwnd_max")
                    .takes_value(true)
                    .required(true)
                    .help("The max cwnd, in packets, for the BIC protocol"),
            )
            .arg(
                Arg::with_name("init_cwnd")
                    .long("init_cwnd")
                    .takes_value(true)
                    .help("Initial cwnd, in packets"),
            )
            .get_matches();

        let cwnd_max = u32::from_str_radix(matches.value_of("cwnd_max").unwrap(), 10)
            .map_err(|e| format!("{:?}", e))?;
        let cwnd_max = cwnd_max * PKTS_TO_BYTES;
        let init_cwnd = if matches.is_present("init_cwnd") {
            u32::from_str_radix(matches.value_of("init_cwnd").unwrap(), 10)
                .map_err(|e| format!("{:?}", e))?
        } else {
            1
        };

        Ok((
            CcpBicDctcpAlg {
                logger: Some(log.clone()),
                cwnd_max,
                init_cwnd,
            },
            String::from(matches.value_of("ipc").unwrap()),
        ))
    }()
    .map_err(|e| warn!(log, "bad argument"; "err" => ?e))
    .unwrap();

    portus::start!(ipc.as_str(), Some(log), cfg).unwrap()
}
