# Calibration data

Instrument settings and calibration values for the Malaspina Strait AZFP deployment
(serial 55158, 67 and 125 kHz, March 2025 – March 2026).

## Files

**`azfp_calibration.csv`**: the values used to convert raw counts to Sv, per month and frequency.
Made by hand from each month's `.XML` file, cross-checked against DFO's Echoview `.ECS` files.

| Column | Source | Cross-checked against |
|---|---|---|
| `EL_dB` | `.XML` | `.ECS` `AzfpEchoLevelMax` |
| `DS` | `.XML` | `.ECS` `AzfpDetectionSlope` |
| `TVR` | `.XML` | `.ECS` `AzfpTransmitVoltageResponse` |
| `VTX` | `.XML` | `.ECS` `AzfpTransmitVoltage` |
| `BP_dB` | `.XML` | `.ECS` `TwoWayBeamAngle` (10·log₁₀ BP) |
| `PL_us` | `.XML` | `.ECS` `PulseDuration` |
| `SS_m_s` | `.ECS` | RBR temperature + pressure |
| `AC_dB_m` | `.ECS` `AbsorptionCoefficient` | — |

The instrument values were identical in the `.XML` and `.ECS` files for all 12 months.

**`azfp_sampling.csv`**: acquisition settings (ping interval, dig rate, range bins, etc.), read from each
month's `.XML`. Distance columns (`sample_spacing_m`, `bin_size_m`, `max_range_m`, `pulse_res_m`,
`lockout_m`) were calculated with the `.XML` header sound speed (1450.5 m/s), so they are about 2% smaller
than with the `.ECS` sound speed. For reference only; not used by the code.

**`azfp_deployment.csv`**: deployment details (serial number, firmware, start time, file counts), read from
each month's `.XML` and `.DPL`. `instrument_depth_m` is the header value (110 m); the cross-check showed the
instrument is at about 117 m. For reference only; not used by the code.

**`azfp_config_report.md`**: summary of the `.XML`/`.DPL` comparison across months.

## Note on March 2026

*Mar 2026: end of deployment (recording stopped 1 Mar 2026 15:08; instrument recovered and reconfigured at
18:36 as 26030118, with identical settings). No .01A files were recorded in March. ECS values break the
seasonal trend: sound speed is 3.2 m/s above February, absorption is above the November peak, and the
commented SourceCal sound speed (1485.68) is copied from February. MSP of 5.41 dbar indicates the pressure
logger was at the surface. These values are not used, and no herring or salmon regions fall in this period.
