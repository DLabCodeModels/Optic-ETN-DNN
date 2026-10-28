
import os
import glob
import csv
import argparse
import random
import zipfile
import json
import hashlib
import io
import warnings

import h5py
import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Global config
# ============================================================

TARGET_HDF5_NAME = "outdoor_night1_data.hdf5"

DEFAULT_SEARCH_ROOTS = [
    "MVSEC_NIGHT",
    "."
]

DEFAULT_RESULT_DIR = "MVSEC_ETN_training_results_stable"

ACTION_NAMES = [
    "GO_FORWARD",
    "SLOW_DOWN",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP"
]

ACTION_TO_ID = {
    "GO_FORWARD": 0,
    "SLOW_DOWN": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "STOP": 4
}


# ============================================================
# 1. Utility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_mvsec_hdf5():
    preferred_names = [
        "outdoor_night1_data.hdf5",
        "outdoor_night2_data.hdf5",
        "outdoor_day1_data.hdf5",
        "outdoor_day2_data.hdf5",
        "indoor_flying1_data.hdf5",
        "indoor_flying2_data.hdf5",
        "indoor_flying3_data.hdf5",
        "indoor_flying4_data.hdf5",
    ]

    for name in preferred_names:
        for root in DEFAULT_SEARCH_ROOTS:
            if not os.path.exists(root):
                continue

            files = glob.glob(
                os.path.join(root, "**", name),
                recursive=True
            )

            files = [
                f for f in files
                if "training_results" not in f
                and "cache" not in f.lower()
                and "result" not in f.lower()
            ]

            if len(files) > 0:
                return files[0]

    h5_files = []

    for root in DEFAULT_SEARCH_ROOTS:
        if os.path.exists(root):
            h5_files.extend(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
            h5_files.extend(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))

    h5_files = [
        f for f in h5_files
        if "training_results" not in f
        and "cache" not in f.lower()
        and "result" not in f.lower()
    ]

    if len(h5_files) > 0:
        return h5_files[0]

    return None


def normalize01(x):
    x = x.astype(np.float32)
    x = x - np.min(x)
    m = np.max(x)

    if m > 1e-8:
        x = x / m

    return x.astype(np.float32)


def robust_normalize01(x, p_low=1.0, p_high=99.0):
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo + 1e-8:
        return normalize01(x)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_uint8_gray(img):
    img = np.asarray(img)

    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[:, :, 0]

    img = img.astype(np.float32)

    if np.max(img) <= 1.5:
        img = img * 255.0
    elif np.max(img) > 255:
        img = normalize01(img) * 255.0

    return np.uint8(np.clip(img, 0, 255))


def resize_gray(gray, height, width):
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def resize_map_nearest(x, height, width):
    return cv2.resize(x, (width, height), interpolation=cv2.INTER_NEAREST)


def colorize_map(x):
    x = normalize01(x)
    x_u8 = np.uint8(x * 255)
    bgr = cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def gray_to_rgb(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def add_title(img_rgb, title):
    img = img_rgb.copy()
    if img.ndim == 2:
        img = gray_to_rgb(img)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    cv2.rectangle(bgr, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_map_csv(path, x):
    np.savetxt(path, x.astype(np.float32), delimiter=",")


# ============================================================
# 2. MVSEC reader
# ============================================================

class MVSECReader:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.f = h5py.File(hdf5_path, "r")

        self.image_ds = self.find_dataset([
            "davis/left/image_raw",
            "/davis/left/image_raw",
            "davis/left/images",
            "left/image_raw"
        ])

        self.image_ts_ds = self.find_dataset([
            "davis/left/image_raw_ts",
            "/davis/left/image_raw_ts",
            "davis/left/image_ts",
            "left/image_raw_ts"
        ])

        self.event_ds = self.find_dataset([
            "davis/left/events",
            "/davis/left/events",
            "left/events"
        ])

        self.event_inds_ds = self.find_dataset([
            "davis/left/image_raw_event_inds",
            "/davis/left/image_raw_event_inds",
            "davis/left/image_event_inds",
            "left/image_raw_event_inds"
        ])

        if self.image_ds is None:
            raise RuntimeError("Cannot find davis/left/image_raw")

        if self.event_ds is None:
            raise RuntimeError("Cannot find davis/left/events")

        if self.event_inds_ds is None:
            raise RuntimeError("Cannot find davis/left/image_raw_event_inds")

        self.num_frames = self.image_ds.shape[0]

        sample = to_uint8_gray(self.image_ds[0])
        self.height, self.width = sample.shape

        print("\nLoaded MVSEC HDF5:")
        print(hdf5_path)
        print(f"Frames: {self.num_frames}")
        print(f"Frame size: {self.width} x {self.height}")
        print(f"Events shape: {self.event_ds.shape}")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def find_dataset(self, candidates):
        for p in candidates:
            p = p[1:] if p.startswith("/") else p
            if p in self.f:
                return self.f[p]
        return None

    def get_frame_delta_s(self, frame_idx, previous_idx, fallback_s=1/30):
        """Use actual image timestamps when present; fallback to camera FPS.

        MVSEC stores timestamps as seconds, but tolerate microsecond units
        when a positive difference exceeds 10 s and is plausible in us.
        """
        if self.image_ts_ds is None:
            return float(fallback_s)
        try:
            t1 = float(np.asarray(self.image_ts_ds[frame_idx]).squeeze())
            t0 = float(np.asarray(self.image_ts_ds[previous_idx]).squeeze())
            dt = t1-t0
            if 0 < dt <= 10:
                return dt
            if 10 < dt <= 1e7:
                return dt * 1e-6
        except (ValueError, TypeError, IndexError):
            pass
        return float(fallback_s)

    def get_frame_time_s(self, frame_idx, fallback_s=None):
        if self.image_ts_ds is not None:
            try:
                t = float(np.asarray(self.image_ts_ds[frame_idx]).squeeze())
                t0 = float(np.asarray(self.image_ts_ds[0]).squeeze())
                delta = t - t0
                step = float(np.asarray(self.image_ts_ds[1]).squeeze()) - t0 if self.num_frames > 1 else 0.0
                if np.isfinite(delta) and delta >= 0:
                    # Determine units from adjacent-frame interval, not absolute elapsed time.
                    # Typical frame interval: ~0.03 seconds or ~33000 microseconds.
                    return delta * 1e-6 if step > 10.0 else delta
            except (IndexError, ValueError, TypeError):
                pass
        return float(fallback_s) if fallback_s is not None else float(frame_idx) / 30.0

    def get_gray(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.num_frames - 1))
        return to_uint8_gray(self.image_ds[frame_idx])

    def get_event_interval(self, frame_idx, max_events=200000, previous_idx=None):
        if frame_idx <= 0:
            start = 0
            end = int(np.asarray(self.event_inds_ds[frame_idx]).squeeze())
        else:
            start_index = frame_idx - 1 if previous_idx is None else previous_idx
            start = int(np.asarray(self.event_inds_ds[start_index]).squeeze())
            end = int(np.asarray(self.event_inds_ds[frame_idx]).squeeze())

        if end <= start:
            return None

        if end - start > max_events:
            start = end - max_events

        return self.event_ds[start:end]


def events_to_binary_maps(events, h, w):
    pos_map = np.zeros((h, w), dtype=np.float32)
    neg_map = np.zeros((h, w), dtype=np.float32)

    if events is None:
        return pos_map, neg_map, 0

    events = np.asarray(events)

    if events.size == 0:
        return pos_map, neg_map, 0

    if events.dtype.names is not None:
        names = events.dtype.names

        x_name = "x" if "x" in names else names[0]
        y_name = "y" if "y" in names else names[1]
        p_name = "p" if "p" in names else (
            "polarity" if "polarity" in names else names[-1]
        )

        xs = events[x_name].astype(np.int32)
        ys = events[y_name].astype(np.int32)
        ps = events[p_name]

    else:
        if events.ndim != 2 or events.shape[1] < 4:
            return pos_map, neg_map, 0

        xs = events[:, 0].astype(np.int32)
        ys = events[:, 1].astype(np.int32)
        ps = events[:, 3]

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    if xs.size == 0:
        return pos_map, neg_map, 0

    if np.min(ps) >= 0:
        pos = ps > 0
        neg = ps <= 0
    else:
        pos = ps > 0
        neg = ps < 0

    pos_map[ys[pos], xs[pos]] = 1.0
    neg_map[ys[neg], xs[neg]] = 1.0

    return pos_map, neg_map, xs.size


# ============================================================
# 3. Experiment-constrained ETN device and feature extractor
# ============================================================
# IMPORTANT EXPERIMENTAL PROVENANCE:
# Arrays below are APPROXIMATELY digitized from the two NIR-intensity
# plots supplied by the user (image readout, not original raw measurements).
# Mode I: positive-decay response; Mode II: polarity-reversal response.
# The intensity range 480-510 uW/um^2 is not experimentally plotted.
# The transition point (495) is ONLY a midpoint assumption for simulation.
# The figure does NOT establish Vg dependence, optical-to-camera calibration,
# device-to-device variability, pulse rise times, or inter-pulse plasticity.

ETN_CALIBRATION_VERSION = "parameter_separation_release_v2"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGURE_POINTS = os.path.join(PROJECT_ROOT, "data", "figure_digitized_points.csv")
DEFAULT_DEVICE_FIT_CONFIG = os.path.join(PROJECT_ROOT, "config", "device_fit.json")
DEFAULT_SIMULATION_CONFIG = os.path.join(PROJECT_ROOT, "config", "simulation.json")


# A single-file convenience fallback for users who copy ETNDNN.py without its data/
# and config/ siblings. These are the exact FIGURE-DIGITIZED EXAMPLE inputs
# shipped in the repository, NOT original measured instrument files.
# User-supplied external files always take precedence. Missing arbitrary custom
# paths are errors; only the three precisely known default assets can fall back.
_EMBEDDED_DEFAULT_ASSETS = {
    os.path.normcase(os.path.abspath(DEFAULT_FIGURE_POINTS)): b'mode,NIR_intensity_uW_um2,signed_peak_V,characteristic_time_us,source_type\r\npositive_decay,270,0.83,70.5,approximate_figure_digitization\r\npositive_decay,300,0.8,69.5,approximate_figure_digitization\r\npositive_decay,330,0.8,66.2,approximate_figure_digitization\r\npositive_decay,360,0.74,66.2,approximate_figure_digitization\r\npositive_decay,390,0.7,59.3,approximate_figure_digitization\r\npositive_decay,420,0.65,47.0,approximate_figure_digitization\r\npositive_decay,450,0.57,33.0,approximate_figure_digitization\r\npositive_decay,480,0.49,18.0,approximate_figure_digitization\r\npolarity_reversal,510,-0.141,45.0,approximate_figure_digitization\r\npolarity_reversal,540,-0.144,39.0,approximate_figure_digitization\r\npolarity_reversal,570,-0.148,37.0,approximate_figure_digitization\r\npolarity_reversal,600,-0.161,34.0,approximate_figure_digitization\r\npolarity_reversal,630,-0.176,33.0,approximate_figure_digitization\r\npolarity_reversal,660,-0.183,30.0,approximate_figure_digitization\r\npolarity_reversal,690,-0.188,20.0,approximate_figure_digitization\r\npolarity_reversal,720,-0.195,13.0,approximate_figure_digitization\r\n',
    os.path.normcase(os.path.abspath(DEFAULT_DEVICE_FIT_CONFIG)): b'{\n  "version": "figure_digitization_v1",\n  "fit_method": "piecewise_linear_interpolation_per_response_mode",\n  "transition_uw_um2": 495.0,\n  "vg": -2.0,\n  "gate_shift_uw_um2_per_v": 0.0,\n  "time_definition_assumption": "reported time is approximated as 1/e time constant",\n  "unmeasured_transition_interval_uw_um2": [\n    480,\n    510\n  ],\n  "extrapolation": "clamp_to_endpoint"\n}\n',
    os.path.normcase(os.path.abspath(DEFAULT_SIMULATION_CONFIG)): b'{\n  "version": "software_simulation_v1",\n  "nir_min_uw_um2": 250.0,\n  "nir_max_uw_um2": 720.0,\n  "fall_threshold": 0.06,\n  "contrast_reference": 0.15,\n  "voltage_threshold_v": 0.04,\n  "trace_time_constant_ms": 150.0,\n  "temporal_time_constant_ms": 80.0,\n  "camera_fps": 30.0,\n  "train_val_ratio": 0.8,\n  "purge_frames": 30,\n  "epochs": 30,\n  "batch_size": 8,\n  "lr": 0.0005,\n  "lambda_action": 0.05,\n  "lambda_focal": 0.3,\n  "base_channels": 24,\n  "risk_rule": "fixed_thresholds",\n  "control_model": "illustrative_kinematic_integration_not_drone_flight",\n  "input_mapping": "grayscale_to_NIR_affine_proxy_not_experimentally_calibrated"\n}\n',
}
_WARNED_EMBEDDED_ASSETS = set()


def _read_project_asset(path):
    requested = os.fspath(path)
    candidates = [requested]
    if not os.path.isabs(requested):
        candidates.append(os.path.join(PROJECT_ROOT, requested))
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "rb") as fh:
                return fh.read()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in _EMBEDDED_DEFAULT_ASSETS:
            if key not in _WARNED_EMBEDDED_ASSETS:
                _WARNED_EMBEDDED_ASSETS.add(key)
                label = os.path.basename(key)
                warnings.warn(
                    "Missing external ETN resource %r; using the matching "
                    "embedded default for standalone execution. The bundled "
                    "NIR measurement table is APPROXIMATELY DIGITIZED FROM "
                    "A FIGURE, NOT RAW EXPERIMENTAL MEASUREMENTS. Put the "
                    "repository data/ and config/ folders beside the script "
                    "or pass explicit CLI resource paths."
                    % label, RuntimeWarning, stacklevel=2)
            return _EMBEDDED_DEFAULT_ASSETS[key]
    raise FileNotFoundError(
        "Required ETN resource not found: %r. Expected data/figure_digitized_points.csv "
        "or config/*.json relative to the script (or supply an existing file via "
        "--figure_points_csv, --device_fit_config, --simulation_config). "
        "Only bundled default assets have an embedded fallback." % requested)


def load_json_config(path):
    return json.loads(_read_project_asset(path).decode("utf-8"))


def file_sha256(path):
    return hashlib.sha256(_read_project_asset(path)).hexdigest()


def load_figure_points(path):
    """Validated tabular parameter input; currently FIGURE-DIGITIZED, not raw data."""
    df = pd.read_csv(io.BytesIO(_read_project_asset(path)))
    cols = {"mode", "NIR_intensity_uW_um2", "signed_peak_V", "characteristic_time_us", "source_type"}
    if not cols.issubset(df.columns) or df.empty:
        raise ValueError("Parameter CSV must include: " + ", ".join(sorted(cols)))
    for k in ("NIR_intensity_uW_um2", "signed_peak_V", "characteristic_time_us"):
        df[k] = pd.to_numeric(df[k], errors="raise")
        if not np.all(np.isfinite(df[k])):
            raise ValueError(f"Non-finite {k}")
    if (df["characteristic_time_us"] <= 0).any():
        raise ValueError("Characteristic time must be positive")
    pos = df[df["mode"] == "positive_decay"].sort_values("NIR_intensity_uW_um2")
    neg = df[df["mode"] == "polarity_reversal"].sort_values("NIR_intensity_uW_um2")
    if len(pos) < 2 or len(neg) < 2 or len(pos) + len(neg) != len(df):
        raise ValueError("Both modes need >=2 points and unambiguous mode names")
    if not (pos["signed_peak_V"] > 0).all() or not (neg["signed_peak_V"] < 0).all():
        raise ValueError("The positive/negative branches must have correct peak signs")
    for branch in (pos, neg):
        if branch["NIR_intensity_uW_um2"].duplicated().any():
            raise ValueError("Duplicate optical intensities within a response branch")
    if max(pos["NIR_intensity_uW_um2"]) >= min(neg["NIR_intensity_uW_um2"]):
        raise ValueError("Branches must be disjoint to define a crossover interval")
    return df, pos, neg


def set_figure_points(path):
    """Load the model source table before creating feature extractors."""
    global FIGURE_POINTS_DF, POS_NIR_UW_UM2, POS_AMPLITUDE_V, POS_DECAY_US
    global NEG_NIR_UW_UM2, NEG_ABS_AMPLITUDE_V, NEG_RECOVERY_US
    FIGURE_POINTS_DF, pos, neg = load_figure_points(path)
    POS_NIR_UW_UM2 = pos["NIR_intensity_uW_um2"].to_numpy(dtype=float)
    POS_AMPLITUDE_V = pos["signed_peak_V"].to_numpy(dtype=float)
    POS_DECAY_US = pos["characteristic_time_us"].to_numpy(dtype=float)
    NEG_NIR_UW_UM2 = neg["NIR_intensity_uW_um2"].to_numpy(dtype=float)
    NEG_ABS_AMPLITUDE_V = -neg["signed_peak_V"].to_numpy(dtype=float)
    NEG_RECOVERY_US = neg["characteristic_time_us"].to_numpy(dtype=float)


set_figure_points(DEFAULT_FIGURE_POINTS)


class CalibratedETNDevice:
    """Effective, experiment-constrained pulse amplitude/tau lookup model.

    Peak voltage in volts and 1/e exponential characteristic time in us.
    Assumes the plotted decay/recovery times can be approximated as 1/e
    constants; the exact experimental fitting definition must be confirmed.
    """
    def __init__(self, transition_uw_um2=495.0, vg=-2.0,
                 reference_vg=-2.0, gate_shift_uw_um2_per_v=0.0):
        self.transition_uw_um2 = float(transition_uw_um2)
        self.vg = float(vg)
        self.reference_vg = float(reference_vg)
        self.gate_shift_uw_um2_per_v = float(gate_shift_uw_um2_per_v)
        pos_edge, neg_edge = float(POS_NIR_UW_UM2[-1]), float(NEG_NIR_UW_UM2[0])
        if not pos_edge <= self.transition_uw_um2 <= neg_edge:
            raise ValueError(f"Assumed crossover must be within measured gap: {pos_edge:g}..{neg_edge:g} uW/um^2")

    @property
    def transition_at_gate(self):
        # Coefficient is 0 by default: not identifiable from attached plots.
        return self.transition_uw_um2 + self.gate_shift_uw_um2_per_v * (self.vg-self.reference_vg)

    def evaluate(self, intensity_uw_um2):
        p = np.asarray(intensity_uw_um2, dtype=np.float64)
        mode_neg = p >= self.transition_at_gate
        ap = np.interp(p, POS_NIR_UW_UM2, POS_AMPLITUDE_V)
        tp = np.interp(p, POS_NIR_UW_UM2, POS_DECAY_US)
        an = np.interp(p, NEG_NIR_UW_UM2, NEG_ABS_AMPLITUDE_V)
        tn = np.interp(p, NEG_NIR_UW_UM2, NEG_RECOVERY_US)
        amplitude_v = np.where(mode_neg, -an, ap)
        tau_us = np.where(mode_neg, tn, tp)
        return amplitude_v, tau_us, mode_neg

    def impulse_response(self, intensity_uw_um2, time_us, onset_us=0.0):
        """Single signed relaxation kernel; NOT a fitted pulse-rise waveform."""
        amp, tau, _ = self.evaluate(float(intensity_uw_um2))
        t = np.asarray(time_us, dtype=float)-float(onset_us)
        return np.where(t >= 0, float(amp)*np.exp(-np.maximum(t,0)/float(tau)), 0.0)

    def illustrative_full_pulse(self, intensity_uw_um2, time_us, onset_us=0,
                                rise_us=40.0, reversal_lag_us=60.0,
                                positive_spike_us=25.0):
        """Shape-only reference to panels b-g (NOT a quantitatively fitted transient).

        The supplied plots contain normalized traces and do not specify the
        optical intensity of each of six stimulus protocols. In reversal mode
        the early *positive* spike amplitude is NOT available in volts. Its
        magnitude is assumed equal to |A-| solely for illustrative waveform
        shape; measured A- and recovery tau determine the negative lobe.
        """
        a, tau, neg = self.evaluate(float(intensity_uw_um2))
        elapsed=np.asarray(time_us,dtype=float)-float(onset_us)
        if not bool(neg):
            rising=np.clip(elapsed/max(rise_us,1e-6),0,1)
            decay=np.exp(-np.maximum(elapsed-rise_us,0)/float(tau))
            return np.where(elapsed>=0,float(a)*rising*decay,0)
        early=np.where(elapsed>=0,abs(float(a))*np.exp(-np.maximum(elapsed,0)/positive_spike_us),0)
        lag=elapsed-reversal_lag_us
        late=np.where(lag>=0,abs(float(a))*np.exp(-np.maximum(lag,0)/float(tau)),0)
        return early-late


def default_etn_params(device_config_path=DEFAULT_DEVICE_FIT_CONFIG,
                       simulation_config_path=DEFAULT_SIMULATION_CONFIG):
    fit = load_json_config(device_config_path)
    sim = load_json_config(simulation_config_path)
    fit_keys = ("vg", "transition_uw_um2", "gate_shift_uw_um2_per_v")
    simulation_keys = ("nir_min_uw_um2", "nir_max_uw_um2", "fall_threshold",
                       "contrast_reference", "voltage_threshold_v",
                       "trace_time_constant_ms", "temporal_time_constant_ms")
    return {**{k: float(fit[k]) for k in fit_keys},
            **{k: float(sim[k]) for k in simulation_keys}}


class ETNFeatureExtractor:
    """7-channel ETN input compatible with the original ETNMotionNet.

    Ch0 grayscale; Ch1 abs(frame change);
    Ch2 NEGATIVE/polarity-reversal response (normalized to parameter-table peak);
    Ch3 POSITIVE-decay response (normalized to parameter-table peak);
    Ch4 cross-frame SOFTWARE motion trace (NOT a 20-70 us device decay);
    Ch5 tau-weighted SOFTWARE temporal evidence (legacy `ppf_state` key);
    Ch6 negative response magnitude, normalized to parameter-table peak.

    The mapping from 8-bit MVSEC intensity -> NIR laser power is a modeling
    PROXY, not a calibration. For physical claims, measured optical intensity
    at each pixel and the true imaging transfer function are required.
    """
    def __init__(self, vg=-2.0, transition_uw_um2=495.0,
                 gate_shift_uw_um2_per_v=0.0,
                 nir_min_uw_um2=250.0, nir_max_uw_um2=720.0,
                 fall_threshold=0.06, contrast_reference=0.15,
                 voltage_threshold_v=0.04,
                 trace_time_constant_ms=150.0,
                 temporal_time_constant_ms=80.0):
        self.device = CalibratedETNDevice(
            transition_uw_um2=transition_uw_um2,
            vg=vg,
            gate_shift_uw_um2_per_v=gate_shift_uw_um2_per_v,
        )
        self.vg = float(vg)
        self.nir_min_uw_um2 = float(nir_min_uw_um2)
        self.nir_max_uw_um2 = float(nir_max_uw_um2)
        self.fall_threshold = float(fall_threshold)
        self.contrast_reference = float(contrast_reference)
        self.voltage_threshold_v = float(voltage_threshold_v)
        self.trace_time_constant_ms = float(trace_time_constant_ms)
        self.temporal_time_constant_ms = float(temporal_time_constant_ms)
        if self.nir_max_uw_um2 <= self.nir_min_uw_um2:
            raise ValueError("nir_max must exceed nir_min")
        if min(self.contrast_reference, self.trace_time_constant_ms,
               self.temporal_time_constant_ms) <= 0:
            raise ValueError("contrast_reference and SOFTWARE tau must be > 0")
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.trace = None
        self.ppf_state = None   # legacy key: software tau-weighted temporal map
        self.last_device_maps = None

    def gate_reversal_threshold(self):
        """Transition on [0,1] CAMERA-PROXY scale, not a measured gate fit."""
        return float(np.clip((self.device.transition_at_gate - self.nir_min_uw_um2) /
                (self.nir_max_uw_um2 - self.nir_min_uw_um2), 0, 1))

    def process(self, gray_u8, dt_s=1.0/30.0):
        gray = np.clip(np.asarray(gray_u8, dtype=np.float32)/255.0, 0, 1)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            self.trace = np.zeros_like(gray)
            self.ppf_state = np.zeros_like(gray)

        dI = gray-self.prev_gray
        abs_dI = np.abs(dI)
        falling_edge = dI < -self.fall_threshold

        # The experimental horizontal axis is optical power density. A camera
        # gray value has NO uniquely defined equivalent in uW/um^2.
        # Adjustable affine proxy retained for MVSEC simulation only.
        power_proxy = self.nir_min_uw_um2 + (
            self.nir_max_uw_um2-self.nir_min_uw_um2
        )*self.prev_gray
        amplitude_v, tau_us, mode_neg = self.device.evaluate(power_proxy)

        # Dataset-specific event contrast scaling; not a device-fit parameter.
        contrast_weight = np.clip(abs_dI/self.contrast_reference, 0.0, 1.0)
        scaled_voltage_v = np.asarray(amplitude_v*contrast_weight, dtype=np.float32)
        active = falling_edge & (np.abs(scaled_voltage_v) >= self.voltage_threshold_v)
        pos_volts = np.where(active & ~mode_neg, scaled_voltage_v, 0).astype(np.float32)
        neg_volts = np.where(active & mode_neg, -scaled_voltage_v, 0).astype(np.float32)

        # Two analog polarity channels, preserving optical-intensity-dependent
        # peak voltage instead of thresholding everything to 0/1.
        # Physical reference values come from the supplied parameter table.
        weak_event = np.clip(pos_volts/max(np.max(POS_AMPLITUDE_V), 1e-8), 0.0, 1.0)
        strong_event = np.clip(neg_volts/max(np.max(NEG_ABS_AMPLITUDE_V), 1e-8), 0.0, 1.0)
        event_energy = np.clip(weak_event+strong_event, 0, 1)

        # Crucial separation of time scales: measured tau=13..71 us. MVSEC
        # frames are separated by milliseconds. Frame-to-frame temporal traces
        # below are a VIDEO/DECODER SOFTWARE integrator, not hardware retention.
        dt_s = max(0.0, float(dt_s))
        a_trace = float(np.exp(-dt_s/(1e-3*self.trace_time_constant_ms)))
        a_temporal = float(np.exp(-dt_s/(1e-3*self.temporal_time_constant_ms)))
        self.trace = np.clip(a_trace*self.trace + event_energy, 0, 1)
        tau_reference_us = max(float(np.max(POS_DECAY_US)), float(np.max(NEG_RECOVERY_US)))
        weighted_tau = event_energy*np.clip(np.asarray(tau_us,dtype=np.float32)/tau_reference_us,0,1)
        self.ppf_state = np.clip(a_temporal*self.ppf_state + weighted_tau, 0, 1)

        # Fixed physical-voltage reference, avoiding per-frame percentile
        # renormalization that would erase amplitude dependence.
        glare_map = strong_event.copy()
        self.last_device_maps = {
            "positive_amplitude_v": pos_volts,
            "negative_magnitude_v": neg_volts,
            "tau_us": np.where(active, tau_us, 0).astype(np.float32),
            "nir_power_proxy_uw_um2": np.asarray(power_proxy,dtype=np.float32),
            "signed_response_v": pos_volts-neg_volts,
            "active_event": active,
        }
        self.prev_gray = gray.copy()

        feat = np.stack([
            gray,
            abs_dI,
            strong_event,
            weak_event,
            self.trace,
            self.ppf_state,
            glare_map,
        ], axis=0).astype(np.float32)
        risk_map, action_id, risk_tuple = make_risk_and_action_from_features(feat)
        return feat, risk_map, action_id, risk_tuple


def save_experimental_etn_calibration(out_dir, config, figure_points_csv=DEFAULT_FIGURE_POINTS):
    """Export approximate digitization, model curves, and independent checks."""
    ensure_dir(out_dir)
    parameter_file = os.path.join(out_dir, "etn_experimental_calibration.json")
    with open(parameter_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": ETN_CALIBRATION_VERSION,
            "source": "Input table; data provenance must be interpreted using source_type per row",
            "figure_points_sha256": file_sha256(figure_points_csv),
            "source_types": sorted(FIGURE_POINTS_DF["source_type"].unique().tolist()),
            "critical_assumptions": [
                "Decay/recovery time approximated as exponential 1/e tau; exact extraction method unverified",
                "Reversal transition assumed 495 uW/um2; 480-510 interval not measured in supplied graph",
                "Gate shift defaults zero because Vg dependence cannot be extracted",
                "Mapping camera grayscale to optical power is an adjustable software proxy, NOT measured equivalence",
                "Inter-frame motion trace is software integration, not measured device retention or fitted PPF",
            ],
            "device_fit_parameters": {k: config[k] for k in ("vg", "transition_uw_um2", "gate_shift_uw_um2_per_v")},
            "simulation_parameters": {k: v for k, v in config.items() if k not in ("vg", "transition_uw_um2", "gate_shift_uw_um2_per_v")},
        }, f, indent=2, ensure_ascii=False)

    FIGURE_POINTS_DF.to_csv(os.path.join(out_dir,"etn_input_parameter_points.csv"), index=False)

    device = CalibratedETNDevice(
        config["transition_uw_um2"],config["vg"],
        gate_shift_uw_um2_per_v=config["gate_shift_uw_um2_per_v"],
    )
    grid=np.linspace(min(config["nir_min_uw_um2"],POS_NIR_UW_UM2[0]),
                     max(config["nir_max_uw_um2"],NEG_NIR_UW_UM2[-1]),471)
    a,t,neg=device.evaluate(grid)
    pd.DataFrame({
        "NIR_intensity_uW_um2":grid,"signed_peak_V":a,
        "characteristic_time_us":t,
        "is_negative_mode":neg.astype(int),
    }).to_csv(os.path.join(out_dir,"etn_calibrated_lookup_curve.csv"),index=False)
    # At each digitized measurement point interpolation must match the input.
    for pp,aa,tt in [(POS_NIR_UW_UM2,POS_AMPLITUDE_V,POS_DECAY_US),
                     (NEG_NIR_UW_UM2,-NEG_ABS_AMPLITUDE_V,NEG_RECOVERY_US)]:
        calc_a,calc_t,_=device.evaluate(pp)
        np.testing.assert_allclose(calc_a,aa,rtol=0,atol=1e-8)
        np.testing.assert_allclose(calc_t,tt,rtol=0,atol=1e-8)

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                         "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    x1=np.linspace(POS_NIR_UW_UM2[0],POS_NIR_UW_UM2[-1],211)
    x2=np.linspace(NEG_NIR_UW_UM2[0],NEG_NIR_UW_UM2[-1],211)
    axs[0,0].scatter(POS_NIR_UW_UM2,POS_DECAY_US,c="#2358a6",s=35,label="Input parameter points (see provenance)")
    axs[0,0].plot(x1,device.evaluate(x1)[1],c="#2358a6",lw=1.3,label="Interpolated model")
    axs[0,0].set(ylabel="Decay time (us)",xlabel="NIR intensity (uW um$^{-2}$)")
    axs[0,1].scatter(POS_NIR_UW_UM2,POS_AMPLITUDE_V,c="#ac252f",s=35,label="Input parameter points (see provenance)")
    axs[0,1].plot(x1,device.evaluate(x1)[0],c="#ac252f",lw=1.3,label="Interpolated model")
    axs[0,1].set(ylabel="Positive amplitude A+ (V)",xlabel="NIR intensity (uW um$^{-2}$)")
    axs[1,0].scatter(NEG_NIR_UW_UM2,NEG_RECOVERY_US,c="#2358a6",s=35,label="Input parameter points (see provenance)")
    axs[1,0].plot(x2,device.evaluate(x2)[1],c="#2358a6",lw=1.3,label="Interpolated model")
    axs[1,0].set(ylabel="Recovery time (us)",xlabel="NIR intensity (uW um$^{-2}$)")
    axs[1,1].scatter(NEG_NIR_UW_UM2,NEG_ABS_AMPLITUDE_V,c="#ac252f",s=35,label="Input parameter points (see provenance)")
    axs[1,1].plot(x2,-device.evaluate(x2)[0],c="#ac252f",lw=1.3,label="Interpolated model")
    axs[1,1].set(ylabel="Negative magnitude |A-| (V)",xlabel="NIR intensity (uW um$^{-2}$)")
    for ax in axs.flat:
        ax.grid(alpha=.14)
    axs[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("ETN input parameters and piecewise interpolation (check provenance)",fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,"etn_experimental_parameter_fit.png"),dpi=250)
    fig.savefig(os.path.join(out_dir,"etn_experimental_parameter_fit.pdf"))
    plt.close(fig)

    # First-order decay kernels only; the 6 measured double-pulse panels do
    # NOT provide per-panel NIR power or raw non-normalized voltage waveforms.
    times=np.linspace(0,800,1601)
    fig,ax=plt.subplots(figsize=(8,3.3))
    for power,col in [(330,"#2358a6"),(450,"#6287c5"),(570,"#c54c56"),(690,"#a51e2a")]:
        y=device.impulse_response(power,times,200)+device.impulse_response(power,times,450)
        ax.plot(times/1000,y,lw=1.6,color=col,label=f"{power} uW/um2")
    ax.axhline(0,c="gray",lw=.6,ls="--")
    ax.set(xlabel="Time (ms)",ylabel="Signed effective response (V)",
           title="Mode-dependent illustrative double-pulse relaxation kernels")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    ax.grid(alpha=.14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,"etn_double_pulse_kernel_illustration.png"),dpi=250)
    fig.savefig(os.path.join(out_dir,"etn_double_pulse_kernel_illustration.pdf"))
    plt.close(fig)

    # Six b-g examples: normalized optical waveforms have pulse-pair ratios,
    # but their intensity/pulse assignment is not specified. Thus the ratios
    # below illustrate shapes only; they are not calibrated physical params.
    fig, axes = plt.subplots(2,3,figsize=(11,5.5),sharex=True,sharey=True)
    examples = [("I-1",330,1.00),("I-2",390,0.60),("I-3",450,1.80),
                ("II-1",540,1.00),("II-2",600,0.65),("II-3",690,1.50)]
    for ax,(label,power,ratio) in zip(axes.flat,examples):
        a1=device.illustrative_full_pulse(power,times,200)
        a2=device.illustrative_full_pulse(power,times,450)*ratio
        z=a1+a2
        z=z/(np.max(np.abs(z))+1e-12)
        ax.plot(times/1000,z,color="#1a4c9c" if label.startswith("I-") else "#cc2a32",lw=1.2)
        ax.axhline(0,color="gray",lw=.5,ls="--")
        ax.set_title(label+f" | {power} uW/um2 (illustrative)",fontsize=9)
        ax.set_xlim(0,.8)
        ax.set_ylim(-1.1,1.1)
        ax.grid(alpha=.12)
    for ax in axes[1,:]: ax.set_xlabel("Time (ms)")
    for ax in axes[:,0]: ax.set_ylabel("Normalized voltage")
    fig.suptitle("Qualitative pulse shapes only: protocol intensities and positive rebound lobes not measured",fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,"etn_six_pulse_shapes_illustrative.png"),dpi=220)
    plt.close(fig)
    print("Exported experimental ETN calibration and validation:",out_dir)


def make_risk_from_features(feat):
    # Fixed-range analog feature combination: per-frame percentile scaling
    # would destroy measured intensity/amplitude trends.
    event_energy = np.clip(feat[2]+feat[3],0,1)
    trace_map = np.clip(feat[4],0,1)
    temporal_map = np.clip(feat[5],0,1)
    negative_magnitude = np.clip(feat[6],0,1)
    risk = 0.50*event_energy+0.35*trace_map+0.10*temporal_map+0.05*negative_magnitude
    return np.clip(risk,0,1).astype(np.float32)


def make_risk_and_action_from_features(feat):
    risk = make_risk_from_features(feat)

    h, w = risk.shape
    lower = risk[int(0.35 * h):, :]

    left = lower[:, :w // 3]
    center = lower[:, w // 3:2 * w // 3]
    right = lower[:, 2 * w // 3:]

    risk_left = float(np.mean(left))
    risk_center = float(np.mean(center))
    risk_right = float(np.mean(right))

    center_minus_side = risk_center - min(risk_left, risk_right)
    lr_diff = risk_left - risk_right
    global_risk = float(np.mean(lower))

    if global_risk > 0.62 and risk_center > 0.48:
        action = "STOP"
    elif risk_center > 0.34 and center_minus_side > 0.035:
        if lr_diff > 0:
            action = "TURN_RIGHT"
        else:
            action = "TURN_LEFT"
    elif risk_center > 0.24:
        action = "SLOW_DOWN"
    else:
        action = "GO_FORWARD"

    return risk, ACTION_TO_ID[action], (risk_left, risk_center, risk_right)


# ============================================================
# 4. Cache generation
# ============================================================

def cache_is_valid(cache_path):
    if not os.path.exists(cache_path):
        return False

    if os.path.getsize(cache_path) < 1024:
        print(f"\nCache too small, remove and rebuild:\n{cache_path}")
        try:
            os.remove(cache_path)
        except Exception:
            pass
        return False

    try:
        data = np.load(cache_path)
        _ = data["X"].shape
        _ = data["Y_event"].shape
        _ = data["Y_action"].shape
        data.close()
        return True

    except Exception as e:
        print("\nInvalid or incomplete cache detected.")
        print(cache_path)
        print(f"Error: {e}")
        print("Removing damaged cache and rebuilding...")

        try:
            os.remove(cache_path)
        except Exception:
            pass

        return False


def build_cache(
    hdf5_path,
    cache_path,
    height=128,
    width=176,
    max_samples=1200,
    frame_step=1,
    max_events=200000,
    rebuild=False,
    etn_params=None,
    camera_fps=30.0,
):
    if not rebuild and cache_is_valid(cache_path):
        print(f"\nUsing existing valid cache:\n{cache_path}")
        return cache_path

    ensure_dir(os.path.dirname(cache_path))

    reader = MVSECReader(hdf5_path)
    etn = ETNFeatureExtractor(**(etn_params or default_etn_params()))

    X_list = []
    Y_event_list = []

    risk_left_list = []
    risk_center_list = []
    risk_right_list = []
    meta_rows = []

    max_frame = reader.num_frames - 1
    count = 0

    print("\nBuilding ETN feature cache...")
    print(f"Target size: {width} x {height}")
    print(f"Max samples: {max_samples}")
    print(f"Frame step: {frame_step}")

    previous_frame_idx = 0
    for frame_idx in range(1, max_frame, frame_step):
        gray = reader.get_gray(frame_idx)
        h0, w0 = gray.shape

        gray_small = resize_gray(gray, height, width)

        events = reader.get_event_interval(
            frame_idx,
            max_events=max_events,
            previous_idx=previous_frame_idx
        )

        pos_map, neg_map, event_count = events_to_binary_maps(events, h0, w0)

        pos_small = resize_map_nearest(pos_map, height, width)
        neg_small = resize_map_nearest(neg_map, height, width)

        dt_s = reader.get_frame_delta_s(
            frame_idx, previous_frame_idx,
            fallback_s=(frame_idx-previous_frame_idx)/float(camera_fps),
        )
        feat, risk_map, rule_action_id, risk_tuple = etn.process(gray_small, dt_s=dt_s)
        previous_frame_idx = frame_idx
        physical_maps = etn.last_device_maps
        active_mask = physical_maps["active_event"]
        active_mean_tau_us = float(np.mean(physical_maps["tau_us"][active_mask])) if np.any(active_mask) else 0.0

        target_event = np.stack([
            pos_small,
            neg_small
        ], axis=0).astype(np.float32)

        X_list.append(feat.astype(np.float16))
        Y_event_list.append(target_event.astype(np.float16))

        risk_left_list.append(risk_tuple[0])
        risk_center_list.append(risk_tuple[1])
        risk_right_list.append(risk_tuple[2])

        frame_time_s = reader.get_frame_time_s(frame_idx, fallback_s=frame_idx/float(camera_fps))
        meta_rows.append([
            count,
            frame_idx,
            event_count,
            rule_action_id,
            ACTION_NAMES[rule_action_id],
            risk_tuple[0],
            risk_tuple[1],
            risk_tuple[2],
            float(np.mean(feat[2] + feat[3])),
            float(np.mean(feat[4])),
            float(np.mean(feat[5])),
            float(np.mean(feat[6])),
            float(np.sum(physical_maps["positive_amplitude_v"])),
            float(np.sum(physical_maps["negative_magnitude_v"])),
            active_mean_tau_us,
            float(np.mean(physical_maps["nir_power_proxy_uw_um2"])),
            dt_s,
            frame_time_s,
            os.path.basename(hdf5_path),
        ])

        count += 1

        if count % 100 == 0:
            print(f"Cached {count} samples...")

        if count >= max_samples:
            break

    reader.close()

    if len(X_list) == 0:
        raise RuntimeError("No samples were cached. Check HDF5 and frame_step/max_samples.")

    X = np.stack(X_list, axis=0)
    Y_event = np.stack(Y_event_list, axis=0)

    # FIXED-THRESHOLD simulation rule labels. No quantile fitted using validation data.
    # These are PSEUDO ACTION TARGETS, never human or flight-controller ground truth.
    Y_action = np.asarray([int(row[3]) for row in meta_rows], dtype=np.int64)


    balanced_meta_rows = []
    for i, row in enumerate(meta_rows):
        balanced_id = int(Y_action[i])
        balanced_name = ACTION_NAMES[balanced_id]
        row2 = list(row)
        row2.insert(5, balanced_id)
        row2.insert(6, balanced_name)
        balanced_meta_rows.append(row2)

    tmp_path = cache_path + ".tmp"
    np.savez_compressed(
        tmp_path,
        X=X,
        Y_event=Y_event,
        Y_action=Y_action
    )

    if os.path.exists(tmp_path + ".npz"):
        tmp_real = tmp_path + ".npz"
    else:
        tmp_real = tmp_path

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass

    os.replace(tmp_real, cache_path)

    meta_path = cache_path.replace(".npz", "_meta.csv")

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cache_index",
            "frame_idx",
            "real_event_count",
            "rule_action_id",
            "rule_action_name",
            "fixed_rule_pseudo_action_id",
            "fixed_rule_pseudo_action_name",
            "risk_left",
            "risk_center",
            "risk_right",
            "etn_event_density",
            "trace_mean",
            "ppf_mean",
            "glare_mean",
            "total_positive_voltage_V",
            "total_negative_magnitude_V",
            "active_mean_tau_us",
            "mean_NIR_power_PROXY_uW_um2",
            "frame_dt_s",
            "frame_time_s",
            "source_sequence_id",
        ])
        writer.writerows(balanced_meta_rows)

    if not cache_is_valid(cache_path):
        raise RuntimeError("Cache was generated but validation failed.")

    print("\nSaved valid training data cache:")
    print(cache_path)
    print(meta_path)
    print(f"X shape: {X.shape}")
    print(f"Y_event shape: {Y_event.shape}")
    print(f"Y_action shape: {Y_action.shape}")

    return cache_path


def load_cache_or_rebuild(cache_path, hdf5_path, args):
    try:
        if not cache_is_valid(cache_path):
            raise RuntimeError("Cache invalid before loading.")

        data = np.load(cache_path)
        num_samples = data["X"].shape[0]
        data.close()
        return num_samples

    except Exception as e:
        print("\nCache loading failed.")
        print(f"Error: {e}")
        print("Rebuilding cache now...")

        try:
            os.remove(cache_path)
        except Exception:
            pass

        build_cache(
            hdf5_path=hdf5_path,
            cache_path=cache_path,
            height=args.height,
            width=args.width,
            max_samples=args.max_samples,
            frame_step=args.frame_step,
            max_events=args.max_events,
            rebuild=True,
            etn_params={k:getattr(args,k) for k in default_etn_params(args.device_fit_config,args.simulation_config)},
            camera_fps=args.camera_fps,
        )

        data = np.load(cache_path)
        num_samples = data["X"].shape[0]
        data.close()
        return num_samples


def save_train_val_indices(out_dir, train_idx, val_idx):
    path = os.path.join(out_dir, "train_val_indices.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "cache_index"])

        for idx in train_idx:
            writer.writerow(["train", int(idx)])

        for idx in val_idx:
            writer.writerow(["val", int(idx)])

    print(f"Saved train/val indices:\n{path}")


# ============================================================
# 5. Dataset
# ============================================================

class ETNMVSECDataset(Dataset):
    def __init__(self, npz_path, indices):
        data = np.load(npz_path)

        self.X = data["X"]
        self.Y_event = data["Y_event"]
        self.Y_action = data["Y_action"]
        self.indices = np.asarray(indices, dtype=np.int64)
        meta_file = npz_path.replace(".npz", "_meta.csv")
        self.metadata = pd.read_csv(meta_file) if os.path.exists(meta_file) else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.X[idx].astype(np.float32)
        y_event = self.Y_event[idx].astype(np.float32)
        y_action = self.Y_action[idx].astype(np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y_event),
            torch.tensor(y_action, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long)
        )


# ============================================================
# 6. Network
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ETNMotionNet(nn.Module):
    def __init__(self, in_ch=7, base=24, num_actions=5):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base, dropout=0.02)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2, dropout=0.03)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4, dropout=0.04)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8, dropout=0.05)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4, dropout=0.03)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout=0.02)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, dropout=0.01)

        self.event_head = nn.Conv2d(base, 2, 1)

        self.action_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(base * 4, num_actions)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        u3 = self.up3(b)
        u3 = self.match_size(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = self.match_size(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = self.match_size(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        event_logits = self.event_head(d1)
        action_logits = self.action_head(b)

        return event_logits, action_logits

    @staticmethod
    def match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x,
                size=ref.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        return x


# ============================================================
# 7. Loss and metrics
# ============================================================

def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)

    inter = torch.sum(prob * target, dim=dims)
    union = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2 * inter + eps) / (union + eps)

    return 1.0 - dice.mean()


def focal_bce_with_logits(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none"
    )

    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    focal = alpha * (1.0 - pt) ** gamma * bce

    return focal.mean()


def event_f1_score(logits, target, threshold=0.50, eps=1e-8):
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = (target > 0.5).float()

    tp = torch.sum(pred * target).item()
    fp = torch.sum(pred * (1.0 - target)).item()
    fn = torch.sum((1.0 - pred) * target).item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def compute_pos_weight(dataset):
    pos = 0.0
    total = 0.0

    for i in range(len(dataset)):
        _, y_event, _, _ = dataset[i]
        pos += float(y_event.sum().item())
        total += float(y_event.numel())

    neg = total - pos

    if pos < 1:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    w = neg / pos
    w = float(np.clip(w, 1.0, 20.0))

    return torch.tensor([w, w], dtype=torch.float32)


def compute_action_class_weight(dataset):
    labels = []

    for i in range(len(dataset)):
        _, _, y_action, _ = dataset[i]
        labels.append(int(y_action.item()))

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(ACTION_NAMES)).astype(np.float32)
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = len(labels) / (np.sum(present) * counts[present])
    weights[present] = np.clip(weights[present], 0.5, 5.0)
    # No examples of absent classes: do not pretend they occurred or score them as trained.

    print("\nAction class counts:", counts.astype(int))
    print("Action class weights:", weights)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 8. Training / evaluation
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    event_bce_loss_fn,
    action_loss_fn,
    device,
    lambda_action=0.05,
    lambda_focal=0.30,
    grad_clip=1.0
):
    model.train()

    total_loss = 0.0
    total_event_loss = 0.0
    total_action_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y_event, y_action, _ in loader:
        x = x.to(device)
        y_event = y_event.to(device)
        y_action = y_action.to(device)

        optimizer.zero_grad()

        event_logits, action_logits = model(x)

        bce = event_bce_loss_fn(event_logits, y_event)
        dice = dice_loss_from_logits(event_logits, y_event)
        focal = focal_bce_with_logits(event_logits, y_event)

        event_loss = bce + dice + lambda_focal * focal
        action_loss = action_loss_fn(action_logits, y_action)
        loss = event_loss + lambda_action * action_loss

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_event_loss += float(event_loss.item()) * x.size(0)
        total_action_loss += float(action_loss.item()) * x.size(0)

        pred_action = torch.argmax(action_logits, dim=1)
        total_correct += int((pred_action == y_action).sum().item())
        total_samples += x.size(0)

    return {
        "loss": total_loss / total_samples,
        "event_loss": total_event_loss / total_samples,
        "action_loss": total_action_loss / total_samples,
        "pseudo_action_agreement": total_correct / total_samples
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    event_bce_loss_fn,
    action_loss_fn,
    device,
    lambda_action=0.05,
    lambda_focal=0.30
):
    model.eval()

    total_loss = 0.0
    total_event_loss = 0.0
    total_action_loss = 0.0
    total_correct = 0
    total_samples = 0

    f1_list = []
    precision_list = []
    recall_list = []

    for x, y_event, y_action, _ in loader:
        x = x.to(device)
        y_event = y_event.to(device)
        y_action = y_action.to(device)

        event_logits, action_logits = model(x)

        bce = event_bce_loss_fn(event_logits, y_event)
        dice = dice_loss_from_logits(event_logits, y_event)
        focal = focal_bce_with_logits(event_logits, y_event)

        event_loss = bce + dice + lambda_focal * focal
        action_loss = action_loss_fn(action_logits, y_action)
        loss = event_loss + lambda_action * action_loss

        total_loss += float(loss.item()) * x.size(0)
        total_event_loss += float(event_loss.item()) * x.size(0)
        total_action_loss += float(action_loss.item()) * x.size(0)

        pred_action = torch.argmax(action_logits, dim=1)
        total_correct += int((pred_action == y_action).sum().item())
        total_samples += x.size(0)

        p, r, f1 = event_f1_score(event_logits, y_event, threshold=0.50)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

    return {
        "loss": total_loss / total_samples,
        "event_loss": total_event_loss / total_samples,
        "action_loss": total_action_loss / total_samples,
        "pseudo_action_agreement": total_correct / total_samples,
        "event_precision": float(np.mean(precision_list)),
        "event_recall": float(np.mean(recall_list)),
        "event_f1": float(np.mean(f1_list))
    }


def plot_training_curves(history, out_dir):
    df = pd.DataFrame(history)

    csv_path = os.path.join(out_dir, "training_metrics.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_event_loss"], label="train event loss")
    plt.plot(df["epoch"], df["val_event_loss"], label="val event loss")
    plt.xlabel("Epoch")
    plt.ylabel("Event loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "event_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_pseudo_action_agreement"], label="train pseudo-label agreement")
    plt.plot(df["epoch"], df["val_pseudo_action_agreement"], label="val pseudo-label agreement")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo-action agreement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pseudo_action_agreement_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["val_event_f1"], label="val event F1")
    plt.plot(df["epoch"], df["val_event_precision"], label="val event precision")
    plt.plot(df["epoch"], df["val_event_recall"], label="val event recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "event_metrics_curve.png"), dpi=300)
    plt.close()

    print(f"Saved training metrics:\n{csv_path}")


# ============================================================
# 9. Visualization
# ============================================================

def make_etn_event_rgb(x_np):
    h, w = x_np.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.uint8(np.clip(x_np[3], 0, 1) * 255)  # positive red
    rgb[:, :, 2] = np.uint8(np.clip(x_np[2], 0, 1) * 255)  # negative blue
    return rgb


def make_real_event_rgb(y_np):
    h, w = y_np.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.uint8(np.clip(y_np[0], 0, 1) * 255)
    rgb[:, :, 2] = np.uint8(np.clip(y_np[1], 0, 1) * 255)
    return rgb


def make_pred_event_rgb(pred_np, threshold=0.50):
    h, w = pred_np.shape[1:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.uint8((pred_np[0] > threshold) * 255)
    rgb[:, :, 2] = np.uint8((pred_np[1] > threshold) * 255)
    return rgb


def make_risk_from_x(x_np):
    risk, action_id, risk_tuple = make_risk_and_action_from_features(x_np)
    return risk, action_id, risk_tuple


@torch.no_grad()
def save_inference_data(
    model,
    dataset,
    device,
    out_dir,
    max_save=200,
    threshold=0.50
):
    model.eval()

    infer_dir = os.path.join(out_dir, "inference_data")
    ensure_dir(infer_dir)

    rows = []

    saved_x = []
    saved_y = []
    saved_pred = []
    saved_indices = []
    saved_action_target = []
    saved_action_pred = []

    for i in range(len(dataset)):
        x, y_event, y_action, cache_idx = dataset[i]
        x_in = x.unsqueeze(0).to(device)

        event_logits, action_logits = model(x_in)
        pred_prob = torch.sigmoid(event_logits)[0].cpu().numpy()
        pred_action = int(torch.argmax(action_logits, dim=1).cpu().item())
        target_action = int(y_action.item())

        y_np = y_event.numpy()
        x_np = x.numpy()

        pred_bin = (pred_prob > threshold).astype(np.float32)
        y_bin = (y_np > 0.5).astype(np.float32)

        tp = float(np.sum(pred_bin * y_bin))
        fp = float(np.sum(pred_bin * (1 - y_bin)))
        fn = float(np.sum((1 - pred_bin) * y_bin))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        risk, rule_action_id, risk_tuple = make_risk_from_x(x_np)

        rows.append([
            int(cache_idx.item()),
            target_action,
            ACTION_NAMES[target_action],
            pred_action,
            ACTION_NAMES[pred_action],
            rule_action_id,
            ACTION_NAMES[rule_action_id],
            precision,
            recall,
            f1,
            float(np.mean(x_np[2] + x_np[3])),
            float(np.mean(x_np[4])),
            float(np.mean(x_np[5])),
            float(np.mean(x_np[6])),
            risk_tuple[0],
            risk_tuple[1],
            risk_tuple[2],
        ])

        if len(saved_x) < max_save:
            saved_x.append(x_np.astype(np.float16))
            saved_y.append(y_np.astype(np.float16))
            saved_pred.append(pred_prob.astype(np.float16))
            saved_indices.append(int(cache_idx.item()))
            saved_action_target.append(target_action)
            saved_action_pred.append(pred_action)

    summary_path = os.path.join(infer_dir, "inference_summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cache_index",
            "target_action_id",
            "target_action_name",
            "pred_action_id",
            "pred_action_name",
            "rule_action_id",
            "rule_action_name",
            "event_precision",
            "event_recall",
            "event_f1",
            "etn_event_density",
            "trace_mean",
            "ppf_mean",
            "glare_mean",
            "risk_left",
            "risk_center",
            "risk_right"
        ])
        writer.writerows(rows)

    npz_path = os.path.join(infer_dir, "inference_maps_subset.npz")

    if len(saved_x) > 0:
        np.savez_compressed(
            npz_path,
            X=np.stack(saved_x, axis=0),
            Y_event=np.stack(saved_y, axis=0),
            Pred_event_prob=np.stack(saved_pred, axis=0),
            cache_indices=np.asarray(saved_indices),
            target_action=np.asarray(saved_action_target),
            pred_action=np.asarray(saved_action_pred),
        )

    print(f"Saved inference summary:\n{summary_path}")
    print(f"Saved inference map subset:\n{npz_path}")


@torch.no_grad()
def save_static_feature_maps(
    model,
    dataset,
    device,
    out_dir,
    num_examples=8,
    threshold=0.50
):
    model.eval()

    static_dir = os.path.join(out_dir, "static_feature_maps")
    ensure_dir(static_dir)

    if len(dataset) == 0:
        return

    chosen = np.linspace(0, len(dataset) - 1, num_examples).astype(int)

    for n, ds_idx in enumerate(chosen):
        x, y_event, y_action, cache_idx = dataset[ds_idx]
        x_in = x.unsqueeze(0).to(device)

        event_logits, action_logits = model(x_in)
        pred_prob = torch.sigmoid(event_logits)[0].cpu().numpy()
        pred_action = int(torch.argmax(action_logits, dim=1).cpu().item())

        x_np = x.numpy()
        y_np = y_event.numpy()

        risk, rule_action_id, risk_tuple = make_risk_from_x(x_np)

        sample_dir = os.path.join(static_dir, f"sample_{n:03d}_cache_{int(cache_idx.item())}")
        ensure_dir(sample_dir)

        np.savez_compressed(
            os.path.join(sample_dir, "all_feature_maps.npz"),
            gray=x_np[0],
            frame_difference=x_np[1],
            etn_strong_event=x_np[2],
            etn_weak_event=x_np[3],
            etn_trace=x_np[4],
            etn_ppf=x_np[5],
            etn_glare=x_np[6],
            risk=risk,
            real_event_pos=y_np[0],
            real_event_neg=y_np[1],
            pred_event_pos_prob=pred_prob[0],
            pred_event_neg_prob=pred_prob[1],
            target_action=int(y_action.item()),
            pred_action=pred_action,
            rule_action=rule_action_id
        )

        save_map_csv(os.path.join(sample_dir, "gray.csv"), x_np[0])
        save_map_csv(os.path.join(sample_dir, "frame_difference.csv"), x_np[1])
        save_map_csv(os.path.join(sample_dir, "etn_trace.csv"), x_np[4])
        save_map_csv(os.path.join(sample_dir, "etn_ppf.csv"), x_np[5])
        save_map_csv(os.path.join(sample_dir, "etn_glare.csv"), x_np[6])
        save_map_csv(os.path.join(sample_dir, "risk.csv"), risk)
        save_map_csv(os.path.join(sample_dir, "pred_event_pos_prob.csv"), pred_prob[0])
        save_map_csv(os.path.join(sample_dir, "pred_event_neg_prob.csv"), pred_prob[1])

        gray = np.uint8(np.clip(x_np[0], 0, 1) * 255)
        dI = np.uint8(np.clip(x_np[1], 0, 1) * 255)

        etn_event_rgb = make_etn_event_rgb(x_np)
        real_event_rgb = make_real_event_rgb(y_np)
        pred_event_rgb = make_pred_event_rgb(pred_prob, threshold=threshold)

        trace_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
        trace_rgb[:, :, 1] = np.uint8(np.clip(x_np[4], 0, 1) * 255)

        ppf_rgb = colorize_map(x_np[5])
        glare_rgb = colorize_map(x_np[6])
        risk_rgb = colorize_map(risk)

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))

        axes[0, 0].imshow(gray, cmap="gray")
        axes[0, 0].set_title("MVSEC grayscale")

        axes[0, 1].imshow(dI, cmap="gray")
        axes[0, 1].set_title("Frame difference")

        axes[0, 2].imshow(etn_event_rgb)
        axes[0, 2].set_title("ETN polarity-selective response")

        axes[0, 3].imshow(trace_rgb)
        axes[0, 3].set_title("ETN motion trace")

        axes[1, 0].imshow(ppf_rgb)
        axes[1, 0].set_title("ETN tau-weighted temporal evidence")

        axes[1, 1].imshow(glare_rgb)
        axes[1, 1].set_title("ETN negative-response magnitude")

        axes[1, 2].imshow(risk_rgb)
        axes[1, 2].set_title("ETN risk map")

        axes[1, 3].imshow(real_event_rgb)
        axes[1, 3].set_title("Real MVSEC event")

        axes[2, 0].imshow(pred_event_rgb)
        axes[2, 0].set_title("Predicted event")

        axes[2, 1].imshow(pred_prob[0], cmap="hot")
        axes[2, 1].set_title("Pred pos prob")

        axes[2, 2].imshow(pred_prob[1], cmap="hot")
        axes[2, 2].set_title("Pred neg prob")

        axes[2, 3].axis("off")
        axes[2, 3].text(0.05, 0.75, f"Target action:\n{ACTION_NAMES[int(y_action.item())]}", fontsize=12)
        axes[2, 3].text(0.05, 0.50, f"Pred action:\n{ACTION_NAMES[pred_action]}", fontsize=12)
        axes[2, 3].text(0.05, 0.25, f"Rule action:\n{ACTION_NAMES[rule_action_id]}", fontsize=12)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig_path = os.path.join(sample_dir, "static_feature_maps.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    print(f"Saved static feature maps:\n{static_dir}")


@torch.no_grad()
def save_feature_video(
    model,
    dataset,
    device,
    out_dir,
    video_frames=200,
    panel_w=260,
    panel_h=180,
    fps=12,
    threshold=0.50
):
    model.eval()

    video_dir = os.path.join(out_dir, "feature_extraction_video")
    ensure_dir(video_dir)

    out_path = os.path.join(video_dir, "etn_feature_extraction_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (panel_w * 4, panel_h * 2))

    n_frames = min(video_frames, len(dataset))

    for i in range(n_frames):
        x, y_event, y_action, cache_idx = dataset[i]
        x_in = x.unsqueeze(0).to(device)

        event_logits, action_logits = model(x_in)
        pred_prob = torch.sigmoid(event_logits)[0].cpu().numpy()
        pred_action = int(torch.argmax(action_logits, dim=1).cpu().item())

        x_np = x.numpy()
        y_np = y_event.numpy()

        risk, rule_action_id, risk_tuple = make_risk_from_x(x_np)

        gray = np.uint8(np.clip(x_np[0], 0, 1) * 255)
        dI = np.uint8(np.clip(x_np[1], 0, 1) * 255)

        gray_rgb = gray_to_rgb(gray)
        di_rgb = gray_to_rgb(dI)
        etn_event_rgb = make_etn_event_rgb(x_np)

        trace_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
        trace_rgb[:, :, 1] = np.uint8(np.clip(x_np[4], 0, 1) * 255)

        real_event_rgb = make_real_event_rgb(y_np)
        pred_event_rgb = make_pred_event_rgb(pred_prob, threshold=threshold)
        glare_rgb = colorize_map(x_np[6])
        risk_rgb = colorize_map(risk)

        panels = [
            add_title(cv2.resize(gray_rgb, (panel_w, panel_h)), "MVSEC frame"),
            add_title(cv2.resize(di_rgb, (panel_w, panel_h)), "Frame difference"),
            add_title(cv2.resize(etn_event_rgb, (panel_w, panel_h)), "ETN event"),
            add_title(cv2.resize(trace_rgb, (panel_w, panel_h)), "ETN trace"),
            add_title(cv2.resize(real_event_rgb, (panel_w, panel_h)), "Real event"),
            add_title(cv2.resize(pred_event_rgb, (panel_w, panel_h)), "Pred event"),
            add_title(cv2.resize(glare_rgb, (panel_w, panel_h)), "Negative response"),
            add_title(cv2.resize(risk_rgb, (panel_w, panel_h)), f"Risk/action: {ACTION_NAMES[pred_action]}")
        ]

        row1 = np.concatenate(panels[:4], axis=1)
        row2 = np.concatenate(panels[4:], axis=1)
        frame_rgb = np.concatenate([row1, row2], axis=0)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(
            frame_bgr,
            f"cache={int(cache_idx.item())} | target={ACTION_NAMES[int(y_action.item())]} | pred={ACTION_NAMES[pred_action]}",
            (10, frame_bgr.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        writer.write(frame_bgr)

    writer.release()
    print(f"Saved feature extraction video:\n{out_path}")


# ============================================================
# 10. 3D plots
# ============================================================

def downsample_for_3d(x, max_h=64, max_w=88):
    h, w = x.shape
    scale = min(max_h / h, max_w / w, 1.0)

    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))

    return cv2.resize(
        x.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def save_3d_surface(map2d, title, out_path):
    z = downsample_for_3d(map2d)
    h, w = z.shape
    xs = np.arange(w)
    ys = np.arange(h)
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, z, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("intensity")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def weighted_centroid(map2d):
    m = map2d.astype(np.float32)
    s = float(np.sum(m))

    if s < 1e-8:
        return np.nan, np.nan, 0.0

    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]

    cx = float(np.sum(xx * m) / s)
    cy = float(np.sum(yy * m) / s)
    energy = float(np.mean(m))

    return cx, cy, energy


def save_3d_feature_plots(dataset, out_dir, sample_count=120, selected_examples=4):
    plot_dir = os.path.join(out_dir, "three_dimensional_feature_plots")
    ensure_dir(plot_dir)

    n = min(sample_count, len(dataset))

    if n <= 2:
        return

    trace_stack = []
    ppf_stack = []
    glare_stack = []
    risk_stack = []
    trajectory_rows = []

    for i in range(n):
        x, _, _, cache_idx = dataset[i]
        x_np = x.numpy()

        risk, action_id, risk_tuple = make_risk_from_x(x_np)

        trace = x_np[4]
        ppf = x_np[5]
        glare = x_np[6]

        trace_stack.append(trace)
        ppf_stack.append(ppf)
        glare_stack.append(glare)
        risk_stack.append(risk)

        cx, cy, energy = weighted_centroid(trace)

        trajectory_rows.append([
            i,
            int(cache_idx.item()),
            cx,
            cy,
            energy,
            action_id,
            ACTION_NAMES[action_id],
            risk_tuple[0],
            risk_tuple[1],
            risk_tuple[2]
        ])

    trace_stack = np.stack(trace_stack, axis=0)
    ppf_stack = np.stack(ppf_stack, axis=0)
    glare_stack = np.stack(glare_stack, axis=0)
    risk_stack = np.stack(risk_stack, axis=0)

    traj_csv = os.path.join(plot_dir, "motion_trace_3d_trajectory.csv")

    with open(traj_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_index",
            "cache_index",
            "centroid_x",
            "centroid_y",
            "trace_energy",
            "rule_action_id",
            "rule_action_name",
            "risk_left",
            "risk_center",
            "risk_right"
        ])
        writer.writerows(trajectory_rows)

    np.savez_compressed(
        os.path.join(plot_dir, "feature_video_3d_data.npz"),
        trace_stack=trace_stack.astype(np.float16),
        ppf_stack=ppf_stack.astype(np.float16),
        glare_stack=glare_stack.astype(np.float16),
        risk_stack=risk_stack.astype(np.float16),
        trajectory=np.asarray(trajectory_rows, dtype=object)
    )

    example_ids = np.linspace(0, n - 1, selected_examples).astype(int)

    for k, idx in enumerate(example_ids):
        save_3d_surface(trace_stack[idx], f"ETN motion trace 3D surface | frame {idx}", os.path.join(plot_dir, f"trace_3d_surface_{k:03d}.png"))
        save_3d_surface(ppf_stack[idx], f"ETN temporal evidence 3D surface | frame {idx}", os.path.join(plot_dir, f"ppf_3d_surface_{k:03d}.png"))
        save_3d_surface(glare_stack[idx], f"ETN negative-response 3D surface | frame {idx}", os.path.join(plot_dir, f"glare_3d_surface_{k:03d}.png"))
        save_3d_surface(risk_stack[idx], f"ETN risk 3D surface | frame {idx}", os.path.join(plot_dir, f"risk_3d_surface_{k:03d}.png"))

    traj = np.asarray(trajectory_rows, dtype=object)
    time_idx = traj[:, 0].astype(float)
    cx = traj[:, 2].astype(float)
    cy = traj[:, 3].astype(float)
    energy = traj[:, 4].astype(float)

    valid = np.isfinite(cx) & np.isfinite(cy)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if np.any(valid):
        ax.plot(cx[valid], cy[valid], time_idx[valid], linewidth=2)
        sc = ax.scatter(cx[valid], cy[valid], time_idx[valid], c=energy[valid], cmap="plasma", s=20)
        fig.colorbar(sc, ax=ax, shrink=0.65, label="trace energy")

    ax.set_title("3D motion trace trajectory from ETN feature video")
    ax.set_xlabel("x centroid")
    ax.set_ylabel("y centroid")
    ax.set_zlabel("time/frame index")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "motion_trace_centroid_3d_trajectory.png"), dpi=300)
    plt.close()

    profiles = []

    for i in range(n):
        trace = trace_stack[i]
        profile = np.mean(trace, axis=0)
        profile = cv2.resize(profile.reshape(1, -1), (88, 1), interpolation=cv2.INTER_AREA).reshape(-1)
        profiles.append(profile)

    profiles = np.stack(profiles, axis=0)
    profiles = normalize01(profiles)

    T, X = np.meshgrid(np.arange(profiles.shape[0]), np.arange(profiles.shape[1]), indexing="ij")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, T, profiles, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_title("3D x-time surface of ETN motion trace video feature")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("time/frame index")
    ax.set_zlabel("trace intensity")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "motion_trace_xtime_3d_surface.png"), dpi=300)
    plt.close()

    print(f"Saved 3D feature plots:\n{plot_dir}")


@torch.no_grad()
def save_action_confusion_matrix(model, dataset, device, out_dir):
    model.eval()

    cm = np.zeros((len(ACTION_NAMES), len(ACTION_NAMES)), dtype=np.int64)

    for i in range(len(dataset)):
        x, _, y_action, _ = dataset[i]
        logits = model(x.unsqueeze(0).to(device))[1]
        pred = int(torch.argmax(logits, dim=1).cpu().item())
        target = int(y_action.item())
        cm[target, pred] += 1

    csv_path = os.path.join(out_dir, "action_confusion_matrix.csv")
    np.savetxt(csv_path, cm, fmt="%d", delimiter=",")

    cm_norm = cm.astype(np.float32)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(ACTION_NAMES)))
    ax.set_yticks(np.arange(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("Rule-derived pseudo action")
    ax.set_title("Pseudo-action agreement (software rules)")

    for i in range(len(ACTION_NAMES)):
        for j in range(len(ACTION_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_confusion_matrix.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(ACTION_NAMES)))
    ax.set_yticks(np.arange(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("Rule-derived pseudo action")
    ax.set_title("Normalized pseudo-action agreement")

    for i in range(len(ACTION_NAMES)):
        for j in range(len(ACTION_NAMES)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_confusion_matrix_normalized.png"), dpi=300)
    plt.close()

    print(f"Saved action confusion matrix:\n{csv_path}")




# ============================================================
# 11. Nature full-story outputs (revised)
# Network architecture + drone scene + 2D/3D features + control
# Every plotted panel has paired CSV data saved in a clear folder.
# ============================================================

NATURE_COLORS = {
    "black": "#222222",
    "gray": "#6E6E6E",
    "light_gray": "#EDEDED",
    "blue": "#2F5D8C",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cream": "#F7F4EA",
    "pale_blue": "#E8F1F8",
    "pale_green": "#E8F5EF",
    "pale_orange": "#FBF1DD",
    "pale_red": "#F8EAE6",
}

ACTION_COLORS_NATURE = {
    0: "#2F5D8C",  # GO_FORWARD
    1: "#E69F00",  # SLOW_DOWN
    2: "#009E73",  # TURN_LEFT
    3: "#D55E00",  # TURN_RIGHT
    4: "#4D4D4D",  # STOP
}

ACTION_SHORT_NAMES = ["GO", "SLOW", "LEFT", "RIGHT", "STOP"]


def set_nature_style_full(font_size=8):
    """Compact, print-friendly Nature-like style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": font_size,
        "axes.linewidth": 0.8,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top",
        color=NATURE_COLORS["black"]
    )


def _safe01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - np.nanmin(x)
    m = np.nanmax(x)
    if m > 1e-8:
        x = x / m
    return np.clip(x, 0.0, 1.0)


def _softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-8)


def _moving_average_1d(x, window=11):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    window = int(max(1, window))
    if window <= 1 or x.size < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, x.size if x.size % 2 == 1 else x.size - 1)
    if window < 3:
        return x.copy()
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values.astype(np.float32)


def _moving_average_2d(x, window=11):
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    for j in range(x.shape[1]):
        y[:, j] = _moving_average_1d(x[:, j], window=window)
    return y


def _clean_short_segments(action_ids, min_len=5):
    """Merge very short action flickers into neighbouring segments without iteration loops."""
    arr = np.asarray(action_ids, dtype=np.int64).copy()
    n = len(arr)
    if n == 0 or min_len <= 1:
        return arr

    for _ in range(2):
        segments = []
        s = 0
        while s < n:
            e = s + 1
            while e < n and arr[e] == arr[s]:
                e += 1
            segments.append((s, e, int(arr[s])))
            s = e

        for idx, (s, e, val) in enumerate(segments):
            if (e - s) >= min_len:
                continue

            prev_seg = segments[idx - 1] if idx > 0 else None
            next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

            if prev_seg is not None and next_seg is not None:
                prev_len = prev_seg[1] - prev_seg[0]
                next_len = next_seg[1] - next_seg[0]
                fill_val = prev_seg[2] if prev_len >= next_len else next_seg[2]
            elif prev_seg is not None:
                fill_val = prev_seg[2]
            elif next_seg is not None:
                fill_val = next_seg[2]
            else:
                fill_val = val

            arr[s:e] = fill_val

    return arr

def _save_dataframe(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    return path


def _long_matrix_dataframe(mat, value_name="value", feature_name=None):
    mat = np.asarray(mat, dtype=np.float32)
    h, w = mat.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    data = {
        "y": yy.reshape(-1),
        "x": xx.reshape(-1),
        value_name: mat.reshape(-1)
    }
    if feature_name is not None:
        data = {"feature": np.repeat(feature_name, h * w), **data}
    return pd.DataFrame(data)


def save_matrix_csv_png(mat, out_prefix, title="", cmap="viridis", colorbar=True):
    """Save a 2D matrix as matrix CSV, long pixel CSV and PNG."""
    ensure_dir(os.path.dirname(out_prefix))
    mat = np.asarray(mat, dtype=np.float32)
    np.savetxt(out_prefix + "_matrix.csv", mat, delimiter=",")
    _long_matrix_dataframe(mat).to_csv(out_prefix + "_pixels.csv", index=False)

    set_nature_style_full(8)
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    im = ax.imshow(mat, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_rgb_csv_png(rgb, out_prefix, title=""):
    """Save an RGB image as RGB pixel CSV and PNG."""
    ensure_dir(os.path.dirname(out_prefix))
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        if np.nanmax(rgb) <= 1.5:
            rgb = np.uint8(np.clip(rgb, 0, 1) * 255)
        else:
            rgb = np.uint8(np.clip(rgb, 0, 255))

    h, w = rgb.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pd.DataFrame({
        "y": yy.reshape(-1),
        "x": xx.reshape(-1),
        "R": rgb[:, :, 0].reshape(-1),
        "G": rgb[:, :, 1].reshape(-1),
        "B": rgb[:, :, 2].reshape(-1)
    }).to_csv(out_prefix + "_rgb_pixels.csv", index=False)

    set_nature_style_full(8)
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_surface_csv_png(z, out_prefix, title="", cmap="viridis", max_h=64, max_w=88):
    """Save a 3D surface as matrix CSV, xyz CSV and PNG."""
    ensure_dir(os.path.dirname(out_prefix))
    z = np.asarray(z, dtype=np.float32)
    np.savetxt(out_prefix + "_surface_matrix.csv", z, delimiter=",")

    h, w = z.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pd.DataFrame({
        "x": xx.reshape(-1),
        "y": yy.reshape(-1),
        "z": z.reshape(-1)
    }).to_csv(out_prefix + "_surface_xyz.csv", index=False)

    scale = min(max_h / h, max_w / w, 1.0)
    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))
    z_small = cv2.resize(z, (new_w, new_h), interpolation=cv2.INTER_AREA)
    yy_s, xx_s = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing="ij")

    set_nature_style_full(8)
    fig = plt.figure(figsize=(3.2, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx_s, yy_s, z_small, cmap=cmap, linewidth=0, antialiased=True, rcount=new_h, ccount=new_w)
    ax.set_title(title, pad=2)
    ax.set_xlabel("x", labelpad=0)
    ax.set_ylabel("y", labelpad=0)
    ax.set_zlabel("value", labelpad=0)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=32, azim=-55)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _make_real_event_rgb_for_story(y_np):
    h, w = y_np[0].shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.uint8(_safe01(y_np[0]) * 255)
    rgb[:, :, 2] = np.uint8(_safe01(y_np[1]) * 255)
    return rgb


def _make_pred_event_rgb_for_story(pred_np):
    h, w = pred_np[0].shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.uint8(_safe01(pred_np[0]) * 255)
    rgb[:, :, 2] = np.uint8(_safe01(pred_np[1]) * 255)
    return rgb


def action_prob_to_drone_command(prob):
    """
    Convert smoothed action probability to high-level drone command readouts.
    These are ETN-driven control readouts, not real flight-controller logs.
    """
    p_go, p_slow, p_left, p_right, p_stop = [float(v) for v in prob]

    forward_velocity = (
        1.00 * p_go
        + 0.42 * p_slow
        + 0.52 * p_left
        + 0.52 * p_right
        + 0.00 * p_stop
    )
    brake = np.clip(0.35 * p_slow + 1.00 * p_stop, 0.0, 1.0)
    yaw_rate_deg_s = 80.0 * (p_left - p_right)
    collective_thrust = (
        0.80 * p_go
        + 0.55 * p_slow
        + 0.62 * p_left
        + 0.62 * p_right
        + 0.15 * p_stop
    )

    return {
        "forward_velocity_cmd": float(np.clip(forward_velocity, 0.0, 1.0)),
        "brake_cmd": float(brake),
        "collective_thrust_cmd": float(np.clip(collective_thrust, 0.0, 1.0)),
        "yaw_rate_cmd_deg_s": float(np.clip(yaw_rate_deg_s, -80.0, 80.0)),
        "body_rate_x_cmd_rad_s": 0.0,
        "body_rate_y_cmd_rad_s": float(-0.50 * np.clip(forward_velocity, 0.0, 1.0)),
        "body_rate_z_cmd_rad_s": float(np.deg2rad(np.clip(yaw_rate_deg_s, -80.0, 80.0)))
    }


# ------------------------- schematic panels -------------------------

def _network_tables(model):
    layer_rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        detail = ""
        if isinstance(module, nn.Conv2d):
            detail = f"in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size}, stride={module.stride}"
        elif isinstance(module, nn.ConvTranspose2d):
            detail = f"in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size}, stride={module.stride}"
        elif isinstance(module, nn.Linear):
            detail = f"in={module.in_features}, out={module.out_features}"
        elif isinstance(module, nn.MaxPool2d):
            detail = f"kernel={module.kernel_size}, stride={module.stride}"
        elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
            detail = f"p={module.p}"
        layer_rows.append([name, module.__class__.__name__, int(params), detail])

    layer_df = pd.DataFrame(layer_rows, columns=["layer_name", "layer_type", "trainable_parameters", "details"])

    nodes = [
        ["input", "7-ch\nETN input", 0.08, 0.58, "pale_blue"],
        ["encoder", "Encoder\nConv + pool", 0.27, 0.58, "pale_blue"],
        ["latent", "Latent\ntemporal state", 0.47, 0.58, "pale_green"],
        ["event_head", "Event head\nevent polarity map", 0.69, 0.72, "pale_blue"],
        ["action_head", "Action head\n5 commands", 0.69, 0.44, "pale_red"],
        ["control", "Control readout\nthrust + body rates", 0.90, 0.44, "pale_orange"],
    ]
    edges = [
        ["input", "encoder", "feature extraction"],
        ["encoder", "latent", "temporal encoding"],
        ["latent", "event_head", "dense prediction"],
        ["latent", "action_head", "global pooling"],
        ["action_head", "control", "command mapping"],
        ["encoder", "event_head", "skip fusion"],
    ]
    node_df = pd.DataFrame(nodes, columns=["node_id", "label", "x", "y", "color_key"])
    edge_df = pd.DataFrame(edges, columns=["source", "target", "meaning"])
    return layer_df, node_df, edge_df


def _draw_network_architecture(ax, node_df, edge_df):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node_dict = {r.node_id: r for r in node_df.itertuples(index=False)}

    for r in edge_df.itertuples(index=False):
        s = node_dict[r.source]
        t = node_dict[r.target]
        rad = 0.0
        if r.meaning == "skip fusion":
            rad = 0.22
        ax.annotate(
            "",
            xy=(t.x - 0.065, t.y),
            xytext=(s.x + 0.065, s.y),
            arrowprops=dict(
                arrowstyle="->",
                lw=0.9,
                color=NATURE_COLORS["gray"],
                connectionstyle=f"arc3,rad={rad}"
            )
        )

    for r in node_df.itertuples(index=False):
        fc = NATURE_COLORS.get(r.color_key, NATURE_COLORS["light_gray"])
        ax.text(
            r.x, r.y, r.label,
            ha="center", va="center",
            fontsize=7.5,
            color=NATURE_COLORS["black"],
            bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=NATURE_COLORS["black"], lw=0.7)
        )

    ax.text(0.02, 0.95, "Network architecture", ha="left", va="top", fontsize=9, fontweight="bold")


def save_network_architecture_diagram(model, out_dir):
    set_nature_style_full(8)
    arch_dir = os.path.join(out_dir, "nature_final_story", "network_architecture")
    ensure_dir(arch_dir)

    layer_df, node_df, edge_df = _network_tables(model)
    _save_dataframe(layer_df, os.path.join(arch_dir, "network_layer_table.csv"))
    _save_dataframe(node_df, os.path.join(arch_dir, "network_architecture_nodes.csv"))
    _save_dataframe(edge_df, os.path.join(arch_dir, "network_architecture_edges.csv"))

    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    _draw_network_architecture(ax, node_df, edge_df)
    fig.tight_layout(pad=0.2)
    png_path = os.path.join(arch_dir, "network_architecture.png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved network architecture:\n{png_path}")
    return png_path, layer_df, node_df, edge_df


def _application_scene_elements():
    elements = [
        ["drone", "Autonomous drone", 0.17, 0.62, 0.10, 0.08],
        ["event_camera", "Event camera / low-light sensor", 0.24, 0.61, 0.03, 0.03],
        ["sensing_cone", "Low-light event sensing field", 0.31, 0.48, 0.33, 0.28],
        ["moving_obstacle", "Moving obstacle", 0.66, 0.55, 0.08, 0.16],
        ["background_object", "Weak-light background object", 0.79, 0.38, 0.11, 0.20],
        ["etn_device", "ETN device dynamics", 0.08, 0.16, 0.16, 0.10],
        ["event_trace", "Event + motion trace", 0.33, 0.16, 0.17, 0.10],
        ["risk_field", "Risk field", 0.58, 0.16, 0.14, 0.10],
        ["control_cmd", "Adaptive command", 0.80, 0.16, 0.15, 0.10],
    ]
    return pd.DataFrame(elements, columns=["element_id", "description", "x", "y", "w", "h"])


def _draw_application_scene(ax, elements_df):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.86, facecolor="#F7F8FA", edgecolor="#D7D7D7", lw=0.8))
    ax.add_patch(plt.Polygon([[0.08, 0.20], [0.94, 0.20], [0.76, 0.43], [0.25, 0.43]],
                             closed=True, facecolor="#E6E8EE", edgecolor="none"))
    for yy in np.linspace(0.23, 0.40, 4):
        ax.plot([0.12, 0.90], [yy, yy], color="white", lw=0.7, alpha=0.9)
    for xx in np.linspace(0.20, 0.82, 5):
        ax.plot([xx, 0.5 + (xx - 0.5) * 0.55], [0.20, 0.43], color="white", lw=0.7, alpha=0.9)

    # sensing cone
    ax.add_patch(plt.Polygon([[0.25, 0.62], [0.64, 0.50], [0.64, 0.78]],
                             closed=True, facecolor=NATURE_COLORS["sky"], edgecolor=NATURE_COLORS["sky"], alpha=0.18))

    # drone
    dx, dy = 0.18, 0.64
    ax.plot([dx - 0.055, dx + 0.055], [dy, dy], color=NATURE_COLORS["blue"], lw=1.3)
    ax.plot([dx, dx], [dy - 0.045, dy + 0.045], color=NATURE_COLORS["blue"], lw=1.3)
    for ox, oy in [(-0.055, 0), (0.055, 0), (0, -0.045), (0, 0.045)]:
        ax.add_patch(plt.Circle((dx + ox, dy + oy), 0.018, facecolor="white", edgecolor=NATURE_COLORS["blue"], lw=0.9))
    ax.scatter([dx], [dy], s=20, color=NATURE_COLORS["blue"], zorder=3)
    ax.text(dx, dy + 0.085, "night-flight drone", ha="center", fontsize=7)

    # objects
    ax.add_patch(plt.Rectangle((0.66, 0.53), 0.075, 0.17, facecolor=NATURE_COLORS["red"], edgecolor="white", lw=0.8, alpha=0.85))
    ax.text(0.70, 0.73, "moving\nobject", ha="center", fontsize=6.5)
    ax.add_patch(plt.Rectangle((0.80, 0.36), 0.11, 0.20, facecolor="#778899", edgecolor="white", lw=0.8, alpha=0.95))
    ax.text(0.855, 0.59, "weak\nbackground", ha="center", fontsize=6.5)

    # event particles
    rng = np.random.default_rng(7)
    ax.scatter(rng.uniform(0.38, 0.64, 42), rng.uniform(0.50, 0.77, 42), s=4, color=NATURE_COLORS["red"], alpha=0.65)
    ax.scatter(rng.uniform(0.43, 0.64, 36), rng.uniform(0.48, 0.72, 36), s=4, color=NATURE_COLORS["blue"], alpha=0.65)

    # processing boxes
    boxes = [
        (0.08, 0.13, 0.16, 0.10, "ETN\ndevice", NATURE_COLORS["pale_green"]),
        (0.33, 0.13, 0.17, 0.10, "event +\ntrace", NATURE_COLORS["pale_blue"]),
        (0.58, 0.13, 0.14, 0.10, "risk\nfield", NATURE_COLORS["pale_orange"]),
        (0.80, 0.13, 0.15, 0.10, "drone\ncommand", NATURE_COLORS["pale_red"]),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=NATURE_COLORS["black"], lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)
    for x1, x2 in [(0.24, 0.33), (0.50, 0.58), (0.72, 0.80)]:
        ax.annotate("", xy=(x2, 0.18), xytext=(x1, 0.18), arrowprops=dict(arrowstyle="->", lw=0.9, color=NATURE_COLORS["gray"]))

    ax.text(0.04, 0.91, "Application scenario", ha="left", va="top", fontsize=9, fontweight="bold")


def save_drone_application_scene(out_dir):
    set_nature_style_full(8)
    scene_dir = os.path.join(out_dir, "nature_final_story", "drone_application_scene")
    ensure_dir(scene_dir)
    elements_df = _application_scene_elements()
    _save_dataframe(elements_df, os.path.join(scene_dir, "application_scene_elements.csv"))

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    _draw_application_scene(ax, elements_df)
    fig.tight_layout(pad=0.2)
    png_path = os.path.join(scene_dir, "drone_application_scene.png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved drone application scene:\n{png_path}")
    return png_path, elements_df


def _closed_loop_tables():
    nodes = [
        ["sensor", "Night image\n/event stream", 0.08, 0.62, "pale_blue"],
        ["etn", "ETN dynamics\nexperiment + trace", 0.27, 0.62, "pale_green"],
        ["network", "ETNMotionNet\nevent + action", 0.49, 0.62, "pale_blue"],
        ["control", "Control readout\nthrust/body rates", 0.71, 0.62, "pale_red"],
        ["drone", "Drone motion\ntrajectory", 0.90, 0.62, "pale_orange"],
        ["environment", "Low-light environment\nobstacle/glare/motion", 0.49, 0.24, "cream"],
    ]
    edges = [
        ["sensor", "etn", "temporal device response"],
        ["etn", "network", "feature tensor"],
        ["network", "control", "action probability"],
        ["control", "drone", "command"],
        ["drone", "environment", "motion update"],
        ["environment", "sensor", "new visual input"],
    ]
    node_df = pd.DataFrame(nodes, columns=["node_id", "label", "x", "y", "color_key"])
    edge_df = pd.DataFrame(edges, columns=["source", "target", "meaning"])
    return node_df, edge_df


def _draw_closed_loop_system(ax, node_df, edge_df):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node_dict = {r.node_id: r for r in node_df.itertuples(index=False)}

    for r in edge_df.itertuples(index=False):
        s = node_dict[r.source]
        t = node_dict[r.target]
        if r.source == "environment" and r.target == "sensor":
            ax.annotate("", xy=(t.x, t.y - 0.10), xytext=(s.x - 0.20, s.y + 0.08),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color=NATURE_COLORS["gray"], connectionstyle="arc3,rad=-0.35"))
        elif r.source == "drone" and r.target == "environment":
            ax.annotate("", xy=(t.x + 0.18, t.y + 0.08), xytext=(s.x, s.y - 0.10),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color=NATURE_COLORS["gray"], connectionstyle="arc3,rad=-0.30"))
        else:
            ax.annotate("", xy=(t.x - 0.065, t.y), xytext=(s.x + 0.065, s.y),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color=NATURE_COLORS["gray"]))

    for r in node_df.itertuples(index=False):
        ax.text(r.x, r.y, r.label, ha="center", va="center", fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.25", fc=NATURE_COLORS.get(r.color_key, "white"), ec=NATURE_COLORS["black"], lw=0.7))
    ax.text(0.02, 0.95, "Closed-loop sensing-to-control pathway", ha="left", va="top", fontsize=9, fontweight="bold")


def save_closed_loop_system_diagram(out_dir):
    set_nature_style_full(8)
    sys_dir = os.path.join(out_dir, "nature_final_story", "closed_loop_system")
    ensure_dir(sys_dir)
    node_df, edge_df = _closed_loop_tables()
    _save_dataframe(node_df, os.path.join(sys_dir, "closed_loop_nodes.csv"))
    _save_dataframe(edge_df, os.path.join(sys_dir, "closed_loop_edges.csv"))

    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    _draw_closed_loop_system(ax, node_df, edge_df)
    fig.tight_layout(pad=0.2)
    png_path = os.path.join(sys_dir, "closed_loop_system_diagram.png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved closed-loop system diagram:\n{png_path}")
    return png_path, node_df, edge_df


# ------------------------- feature exports -------------------------

@torch.no_grad()
def save_all_2d_3d_feature_outputs(model, dataset, device, out_dir, num_examples=6, threshold=0.50):
    """Save key 2D/3D feature data as CSV + PNG for selected examples."""
    set_nature_style_full(8)
    feature_dir = os.path.join(out_dir, "nature_final_story", "feature_2d_3d_outputs")
    ensure_dir(feature_dir)
    model.eval()

    if len(dataset) == 0:
        return None

    # Nature figure should show key evidence, not hundreds of redundant maps.
    num_selected = max(1, min(int(num_examples), 3, len(dataset)))
    chosen = np.linspace(0, len(dataset) - 1, num_selected).astype(int)
    summary_rows = []
    representative = None
    best_rep_score = -1.0

    for n, ds_idx in enumerate(chosen):
        x, y_event, y_action, cache_idx = dataset[int(ds_idx)]
        x_np = x.numpy()
        y_np = y_event.numpy()

        event_logits, action_logits = model(x.unsqueeze(0).to(device))
        pred_prob = torch.sigmoid(event_logits)[0].cpu().numpy()
        action_prob = _softmax_np(action_logits[0].cpu().numpy())
        pred_action = int(np.argmax(action_prob))
        risk, rule_action_id, risk_tuple = make_risk_from_x(x_np)

        etn_event_energy = _safe01(x_np[2] + x_np[3])
        real_event_energy = _safe01(y_np[0] + y_np[1])
        pred_event_energy = _safe01(pred_prob[0] + pred_prob[1])

        sample_dir = os.path.join(feature_dir, f"sample_{n:03d}_cache_{int(cache_idx.item())}")
        ensure_dir(sample_dir)

        key_maps = {
            "gray_frame": x_np[0],
            "etn_event_energy": etn_event_energy,
            "motion_trace": x_np[4],
            "risk_map": risk,
            "real_event_energy": real_event_energy,
            "pred_event_energy": pred_event_energy,
        }

        # One compact CSV containing all key 2D maps for this sample.
        long_key_maps = []
        for name, mat in key_maps.items():
            save_matrix_csv_png(
                mat,
                os.path.join(sample_dir, name),
                title=name.replace("_", " "),
                cmap="viridis",
                colorbar=True
            )
            save_surface_csv_png(
                mat,
                os.path.join(sample_dir, name + "_3d"),
                title=name.replace("_", " ") + " 3D",
                cmap="viridis"
            )
            long_key_maps.append(_long_matrix_dataframe(mat, value_name="value", feature_name=name))

        pd.concat(long_key_maps, ignore_index=True).to_csv(
            os.path.join(sample_dir, "key_2d_maps_long_pixels.csv"), index=False
        )

        # RGB event summaries.
        etn_rgb = make_etn_event_rgb(x_np)
        real_rgb = _make_real_event_rgb_for_story(y_np)
        pred_rgb = _make_pred_event_rgb_for_story(pred_prob)
        save_rgb_csv_png(etn_rgb, os.path.join(sample_dir, "etn_event_rgb"), title="ETN signed-response RGB")
        save_rgb_csv_png(real_rgb, os.path.join(sample_dir, "real_event_rgb"), title="Real event RGB")
        save_rgb_csv_png(pred_rgb, os.path.join(sample_dir, "pred_event_rgb"), title="Predicted event RGB")

        rep_score = float(risk_tuple[1] + np.mean(x_np[4]) + np.mean(etn_event_energy))
        if rep_score > best_rep_score:
            best_rep_score = rep_score
            representative = {
                "x_np": x_np,
                "y_np": y_np,
                "pred_prob": pred_prob,
                "risk": risk,
                "action_prob": action_prob,
                "pred_action": pred_action,
                "cache_idx": int(cache_idx.item())
            }

        summary_rows.append([
            n, int(ds_idx), int(cache_idx.item()),
            int(y_action.item()), ACTION_NAMES[int(y_action.item())],
            pred_action, ACTION_NAMES[pred_action],
            float(action_prob[0]), float(action_prob[1]), float(action_prob[2]), float(action_prob[3]), float(action_prob[4]),
            float(np.mean(etn_event_energy)), float(np.mean(x_np[4])), float(np.mean(x_np[5])), float(np.mean(x_np[6])),
            float(risk_tuple[0]), float(risk_tuple[1]), float(risk_tuple[2])
        ])

    summary_df = pd.DataFrame(summary_rows, columns=[
        "sample_id", "dataset_index", "cache_index",
        "target_action_id", "target_action_name",
        "pred_action_id", "pred_action_name",
        "p_GO_FORWARD", "p_SLOW_DOWN", "p_TURN_LEFT", "p_TURN_RIGHT", "p_STOP",
        "etn_event_energy_mean", "trace_mean", "ppf_mean", "glare_mean",
        "risk_left", "risk_center", "risk_right"
    ])
    _save_dataframe(summary_df, os.path.join(feature_dir, "selected_feature_summary.csv"))
    print(f"Saved key 2D/3D feature CSV and PNG:\n{feature_dir}")
    return representative


# ------------------------- command and trajectory exports -------------------------

@torch.no_grad()
def save_drone_command_and_trajectory_outputs(model, dataset, device, out_dir, max_frames=450, dt=1.0 / 30.0, threshold=0.50):
    """
    Save command curves, command log, 2D trajectory and 3D trajectory.
    Raw probabilities are preserved; plotted commands use smoothed probabilities.
    """
    set_nature_style_full(8)
    control_dir = os.path.join(out_dir, "nature_final_story", "drone_control_outputs")
    ensure_dir(control_dir)
    model.eval()

    if dataset.metadata is None:
        raise RuntimeError("Trajectory generation requires per-frame metadata with sequence ID and timestamps")
    selected_meta = dataset.metadata.iloc[dataset.indices].copy()
    chosen_sequence = str(selected_meta["source_sequence_id"].iloc[0])
    selected_meta = selected_meta[selected_meta["source_sequence_id"].astype(str) == chosen_sequence]
    selected_meta = selected_meta.sort_values("frame_idx", kind="stable").head(max_frames)
    if len(selected_meta) < 2:
        raise ValueError("Trajectory requires at least two consecutive samples of one sequence")
    chosen_cache_indices = selected_meta["cache_index"].astype(int).tolist()
    dataset_lookup = {int(cache_idx):i for i,cache_idx in enumerate(dataset.indices)}
    frame_times = selected_meta["frame_time_s"].to_numpy(dtype=float)
    frame_times = frame_times - frame_times[0]
    if np.any(np.diff(frame_times) <= 0):
        raise ValueError("Non-monotonic timestamps; do not fabricate a navigation trajectory")
    n = len(chosen_cache_indices)
    rows = []

    for i, chosen_cache_index in enumerate(chosen_cache_indices):
        x, y_event, y_action, cache_idx = dataset[dataset_lookup[chosen_cache_index]]
        x_np = x.numpy()
        y_np = y_event.numpy()

        event_logits, action_logits = model(x.unsqueeze(0).to(device))
        pred_event = torch.sigmoid(event_logits)[0].cpu().numpy()
        action_prob = _softmax_np(action_logits[0].cpu().numpy())
        pred_action_raw = int(np.argmax(action_prob))
        risk, rule_action_id, risk_tuple = make_risk_from_x(x_np)

        pred_bin = (pred_event > threshold).astype(np.float32)
        y_bin = (y_np > 0.5).astype(np.float32)
        tp = float(np.sum(pred_bin * y_bin))
        fp = float(np.sum(pred_bin * (1.0 - y_bin)))
        fn = float(np.sum((1.0 - pred_bin) * y_bin))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        rows.append({
            "frame": i,
            "time_s": float(frame_times[i]),
            "cache_index": int(cache_idx.item()),
            "target_action_id": int(y_action.item()),
            "target_action_name": ACTION_NAMES[int(y_action.item())],
            "pred_action_id_raw": pred_action_raw,
            "pred_action_name_raw": ACTION_NAMES[pred_action_raw],
            "rule_action_id": int(rule_action_id),
            "rule_action_name": ACTION_NAMES[int(rule_action_id)],
            "p_raw_GO_FORWARD": float(action_prob[0]),
            "p_raw_SLOW_DOWN": float(action_prob[1]),
            "p_raw_TURN_LEFT": float(action_prob[2]),
            "p_raw_TURN_RIGHT": float(action_prob[3]),
            "p_raw_STOP": float(action_prob[4]),
            "risk_left_raw": float(risk_tuple[0]),
            "risk_center_raw": float(risk_tuple[1]),
            "risk_right_raw": float(risk_tuple[2]),
            "risk_global_raw": float(np.mean(risk)),
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": f1,
        })

    df = pd.DataFrame(rows)
    df["source_sequence_id"] = chosen_sequence
    df["trajectory_type"] = "illustrative_kinematic_software_simulation"

    # Smooth probabilities and risk for plotting/command readout.
    p_raw = df[[f"p_raw_{name}" for name in ACTION_NAMES]].values.astype(np.float32)
    p_smooth = _moving_average_2d(p_raw, window=11)
    p_smooth = p_smooth / (np.sum(p_smooth, axis=1, keepdims=True) + 1e-8)

    raw_action = np.argmax(p_raw, axis=1).astype(np.int64)
    smooth_action = np.argmax(p_smooth, axis=1).astype(np.int64)
    clean_action = _clean_short_segments(smooth_action, min_len=5)

    for j, name in enumerate(ACTION_NAMES):
        df[f"p_{name}"] = p_raw[:, j]  # compatibility: raw probability
        df[f"p_smooth_{name}"] = p_smooth[:, j]

    df["pred_action_id"] = clean_action
    df["pred_action_name"] = [ACTION_NAMES[int(a)] for a in clean_action]
    df["pred_action_id_smooth_argmax"] = smooth_action
    df["pred_action_name_smooth_argmax"] = [ACTION_NAMES[int(a)] for a in smooth_action]

    for col in ["risk_left_raw", "risk_center_raw", "risk_right_raw", "risk_global_raw"]:
        df[col.replace("_raw", "_smooth")] = _moving_average_1d(df[col].values, window=11)

    # Use smoothed probabilities for command readout.
    command_rows = [action_prob_to_drone_command(p_smooth[i]) for i in range(len(df))]
    cmd_df = pd.DataFrame(command_rows)
    for col in cmd_df.columns:
        df[col] = cmd_df[col].values

    # Compatibility aliases used by older code/plots.
    df["risk_left"] = df["risk_left_raw"]
    df["risk_center"] = df["risk_center_raw"]
    df["risk_right"] = df["risk_right_raw"]
    df["risk_global"] = df["risk_global_raw"]

    for col in [
        "forward_velocity_cmd", "brake_cmd", "collective_thrust_cmd",
        "yaw_rate_cmd_deg_s", "body_rate_x_cmd_rad_s", "body_rate_y_cmd_rad_s", "body_rate_z_cmd_rad_s"
    ]:
        df[col + "_abs_diff"] = df[col].diff().abs().fillna(0)

    df["action_switch"] = (df["pred_action_id"].diff().fillna(0).abs() > 0).astype(int)

    # Integrate an illustrative command-driven trajectory.
    x_pos, y_pos, z_pos, headings = [0.0], [0.0], [1.2], [0.0]
    heading = 0.0
    for i in range(1, len(df)):
        v = float(df.loc[i, "forward_velocity_cmd"]) * (1.0 - 0.80 * float(df.loc[i, "brake_cmd"]))
        v = max(v, 0.0) * 2.6
        yaw = float(df.loc[i, "body_rate_z_cmd_rad_s"])
        thrust = float(df.loc[i, "collective_thrust_cmd"])
        dt_i = float(frame_times[i] - frame_times[i-1])
        heading = heading + yaw * dt_i
        x_pos.append(x_pos[-1] + v * np.cos(heading) * dt_i)
        y_pos.append(y_pos[-1] + v * np.sin(heading) * dt_i)
        z_new = z_pos[-1] + 0.15 * (thrust - 0.55) * dt_i - 0.07 * float(df.loc[i, "brake_cmd"]) * dt_i
        z_pos.append(float(np.clip(z_new, 0.6, 2.2)))
        headings.append(heading)

    df["traj_x_m"] = x_pos
    df["traj_y_m"] = y_pos
    df["traj_z_m"] = z_pos
    df["heading_rad"] = headings

    full_log_csv = os.path.join(control_dir, "drone_command_log.csv")
    _save_dataframe(df, full_log_csv)

    # Panel E data: probability heatmap.
    panel_e_cols = ["frame", "time_s", "pred_action_id_raw", "pred_action_name_raw", "pred_action_id", "pred_action_name"]
    for name in ACTION_NAMES:
        panel_e_cols += [f"p_raw_{name}", f"p_smooth_{name}"]
    panel_e_df = df[panel_e_cols].copy()
    _save_dataframe(panel_e_df, os.path.join(control_dir, "panel_e_action_probabilities_raw_and_smoothed.csv"))

    # Panel F data: control commands and risk.
    panel_f_cols = [
        "frame", "time_s",
        "risk_left_raw", "risk_center_raw", "risk_right_raw",
        "risk_left_smooth", "risk_center_smooth", "risk_right_smooth",
        "forward_velocity_cmd", "collective_thrust_cmd", "brake_cmd",
        "yaw_rate_cmd_deg_s", "body_rate_z_cmd_rad_s",
        "pred_action_id", "pred_action_name"
    ]
    panel_f_df = df[panel_f_cols].copy()
    _save_dataframe(panel_f_df, os.path.join(control_dir, "panel_f_control_commands_and_risk.csv"))

    # Panel G data: trajectory.
    panel_g_cols = [
        "frame", "time_s", "traj_x_m", "traj_y_m", "traj_z_m", "heading_rad",
        "pred_action_id", "pred_action_name", "risk_center_smooth",
        "forward_velocity_cmd", "collective_thrust_cmd", "brake_cmd", "yaw_rate_cmd_deg_s"
    ]
    panel_g_df = df[panel_g_cols].copy()
    _save_dataframe(panel_g_df, os.path.join(control_dir, "panel_g_trajectory_2d_3d.csv"))
    _save_dataframe(panel_g_df, os.path.join(control_dir, "drone_motion_trace_2d.csv"))
    _save_dataframe(panel_g_df, os.path.join(control_dir, "drone_motion_trace_3d.csv"))

    # Metrics.
    eps = 1e-8
    risk_center = df["risk_center_smooth"].values.astype(np.float32)
    brake = df["brake_cmd"].values.astype(np.float32)
    yaw = df["yaw_rate_cmd_deg_s"].values.astype(np.float32)
    lr_risk = (df["risk_left_smooth"] - df["risk_right_smooth"]).values.astype(np.float32)
    risk_brake_corr = float(np.corrcoef(risk_center, brake)[0, 1]) if np.std(risk_center) > eps and np.std(brake) > eps else 0.0
    lr_yaw_corr = float(np.corrcoef(lr_risk, yaw)[0, 1]) if np.std(lr_risk) > eps and np.std(yaw) > eps else 0.0
    action_switch_rate = float(df["action_switch"].mean())
    mean_event_f1 = float(df["event_f1"].mean())
    command_smoothness = float(df[[
        "forward_velocity_cmd_abs_diff", "brake_cmd_abs_diff", "collective_thrust_cmd_abs_diff", "yaw_rate_cmd_deg_s_abs_diff"
    ]].mean().mean())
    risk_brake_alignment = float(np.clip(risk_brake_corr, 0, 1))
    turn_alignment = float(np.clip(abs(lr_yaw_corr), 0, 1))
    command_stability = float(1.0 - np.clip(action_switch_rate, 0, 1))
    control_consistency_index = float(0.30 * mean_event_f1 + 0.25 * risk_brake_alignment + 0.25 * turn_alignment + 0.20 * command_stability)

    metrics = {
        "mean_event_f1": mean_event_f1,
        "risk_brake_correlation_raw": risk_brake_corr,
        "risk_brake_alignment_plot": risk_brake_alignment,
        "left_right_risk_yaw_correlation_raw": lr_yaw_corr,
        "turn_alignment_plot": turn_alignment,
        "action_switch_rate": action_switch_rate,
        "command_stability_plot": command_stability,
        "command_smoothness": command_smoothness,
        "control_consistency_index": control_consistency_index,
        "mean_risk_center": float(df["risk_center_smooth"].mean()),
        "mean_collective_thrust_cmd": float(df["collective_thrust_cmd"].mean()),
        "mean_brake_cmd": float(df["brake_cmd"].mean()),
    }

    panel_h_df = pd.DataFrame([
        ["Event F1", metrics["mean_event_f1"], metrics["mean_event_f1"], "Mean event-map F1 over plotted frames"],
        ["Risk-brake alignment", metrics["risk_brake_correlation_raw"], metrics["risk_brake_alignment_plot"], "Correlation between centre risk and brake command; negative values clipped only for plotting"],
        ["Turn-risk alignment", metrics["left_right_risk_yaw_correlation_raw"], metrics["turn_alignment_plot"], "Absolute correlation between left-right risk imbalance and yaw-rate command"],
        ["Command stability", 1.0 - metrics["action_switch_rate"], metrics["command_stability_plot"], "One minus the smoothed action switching rate"],
        ["Control consistency index", metrics["control_consistency_index"], metrics["control_consistency_index"], "Illustrative software-defined control consistency index; not flight success"],
    ], columns=["metric", "raw_value", "plot_value", "definition"])

    _save_dataframe(pd.DataFrame([metrics]), os.path.join(control_dir, "drone_control_metrics.csv"))
    _save_dataframe(panel_h_df, os.path.join(control_dir, "panel_h_control_metrics.csv"))

    # PNG: probability heatmap.
    _plot_action_probability_heatmap(panel_e_df, os.path.join(control_dir, "panel_e_action_probability_heatmap.png"))

    # PNG: command curves.
    _plot_command_modulation(panel_f_df, os.path.join(control_dir, "drone_command_variation.png"))

    # PNG: 2D and 3D trajectories.
    _plot_trajectory_2d(panel_g_df, os.path.join(control_dir, "drone_motion_trace_2d.png"))
    _plot_trajectory_3d(panel_g_df, os.path.join(control_dir, "drone_motion_trace_3d.png"))

    # PNG: metrics.
    _plot_control_metrics(panel_h_df, os.path.join(control_dir, "drone_control_metrics.png"))

    print(f"Saved drone command log, panel CSVs and trajectory outputs:\n{control_dir}")
    return df, metrics


def _plot_action_probability_heatmap(panel_e_df, out_png):
    set_nature_style_full(8)
    t = panel_e_df["time_s"].values
    p = panel_e_df[[f"p_smooth_{name}" for name in ACTION_NAMES]].values.T
    fig, ax = plt.subplots(figsize=(4.8, 2.15))
    extent = [t.min(), t.max(), -0.5, len(ACTION_NAMES) - 0.5]
    im = ax.imshow(p, aspect="auto", origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(ACTION_NAMES)))
    ax.set_yticklabels(ACTION_SHORT_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action")
    ax.set_title("Smoothed action probability")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Probability", rotation=90)
    cb.ax.tick_params(labelsize=6, width=0.6, length=2)
    _despine(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_action_ribbon(ax, t, action_ids, height=0.12):
    y0, y1 = 0.0, height
    ax.set_ylim(0, height)
    ax.set_xlim(t.min(), t.max())
    ax.axis("off")
    for i in range(1, len(t)):
        aid = int(action_ids[i])
        ax.axvspan(t[i - 1], t[i], ymin=0, ymax=1, color=ACTION_COLORS_NATURE.get(aid, "k"), lw=0)


def _plot_command_modulation(panel_f_df, out_png):
    set_nature_style_full(8)
    t = panel_f_df["time_s"].values
    fig = plt.figure(figsize=(5.2, 3.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.0, 0.18], hspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, panel_f_df["risk_center_smooth"], color=NATURE_COLORS["black"], lw=1.1, label="centre")
    ax1.plot(t, panel_f_df["risk_left_smooth"], color=NATURE_COLORS["blue"], lw=0.75, alpha=0.8, label="left")
    ax1.plot(t, panel_f_df["risk_right_smooth"], color=NATURE_COLORS["green"], lw=0.75, alpha=0.8, label="right")
    ax1.set_ylabel("Risk")
    ax1.set_title("Risk and command modulation")
    ax1.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.5)
    _despine(ax1)

    ax2.plot(t, panel_f_df["forward_velocity_cmd"], color=NATURE_COLORS["blue"], lw=1.1, label="forward")
    ax2.plot(t, panel_f_df["collective_thrust_cmd"], color=NATURE_COLORS["orange"], lw=1.1, label="thrust")
    ax2.plot(t, panel_f_df["brake_cmd"], color=NATURE_COLORS["red"], lw=1.1, label="brake")
    ax2.set_ylabel("Norm. command")
    ax2.set_ylim(-0.04, 1.05)
    ax2b = ax2.twinx()
    ax2b.plot(t, panel_f_df["yaw_rate_cmd_deg_s"], color=NATURE_COLORS["purple"], lw=0.95, ls="--", label="yaw")
    ax2b.set_ylabel("Yaw (deg s$^{-1}$)")
    ax2b.spines["top"].set_visible(False)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=4, loc="upper right", handlelength=1.5)
    _despine(ax2)

    _plot_action_ribbon(ax3, t, panel_f_df["pred_action_id"].values, height=1)
    ax2.set_xlabel("Time (s)")
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.14, top=0.90, hspace=0.34)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_2d(panel_g_df, out_png):
    set_nature_style_full(8)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i in range(1, len(panel_g_df)):
        aid = int(panel_g_df.loc[i, "pred_action_id"])
        ax.plot(
            panel_g_df.loc[i - 1:i, "traj_x_m"],
            panel_g_df.loc[i - 1:i, "traj_y_m"],
            lw=1.2,
            color=ACTION_COLORS_NATURE.get(aid, NATURE_COLORS["black"])
        )
    ax.scatter(panel_g_df["traj_x_m"].iloc[0], panel_g_df["traj_y_m"].iloc[0], s=20, color=NATURE_COLORS["black"], marker="o", zorder=3)
    ax.scatter(panel_g_df["traj_x_m"].iloc[-1], panel_g_df["traj_y_m"].iloc[-1], s=35, color=NATURE_COLORS["black"], marker="*", zorder=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Command-coloured trajectory")
    ax.axis("equal")
    _despine(ax)
    handles = [plt.Line2D([0], [0], color=ACTION_COLORS_NATURE[i], lw=1.8) for i in range(len(ACTION_NAMES))]
    ax.legend(handles, ACTION_SHORT_NAMES, frameon=False, ncol=2, loc="best", handlelength=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_3d(panel_g_df, out_png):
    set_nature_style_full(8)
    fig = plt.figure(figsize=(3.4, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, len(panel_g_df)):
        aid = int(panel_g_df.loc[i, "pred_action_id"])
        ax.plot(
            panel_g_df.loc[i - 1:i, "traj_x_m"],
            panel_g_df.loc[i - 1:i, "traj_y_m"],
            panel_g_df.loc[i - 1:i, "traj_z_m"],
            lw=1.1,
            color=ACTION_COLORS_NATURE.get(aid, NATURE_COLORS["black"])
        )
    ax.scatter(panel_g_df["traj_x_m"].iloc[0], panel_g_df["traj_y_m"].iloc[0], panel_g_df["traj_z_m"].iloc[0], s=20, color=NATURE_COLORS["black"], marker="o")
    ax.scatter(panel_g_df["traj_x_m"].iloc[-1], panel_g_df["traj_y_m"].iloc[-1], panel_g_df["traj_z_m"].iloc[-1], s=35, color=NATURE_COLORS["black"], marker="*")
    ax.set_xlabel("x (m)", labelpad=0)
    ax.set_ylabel("y (m)", labelpad=0)
    ax.set_zlabel("z (m)", labelpad=0)
    ax.set_title("3D trajectory")
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_control_metrics(panel_h_df, out_png):
    set_nature_style_full(8)
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    x = np.arange(len(panel_h_df))
    vals = panel_h_df["plot_value"].values.astype(float)
    bar_colors = [NATURE_COLORS["blue"], NATURE_COLORS["red"], NATURE_COLORS["green"], NATURE_COLORS["gray"], NATURE_COLORS["purple"]]
    ax.bar(x, vals, color=bar_colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(["Event\nF1", "Risk\nbrake", "Risk\nyaw", "Stability", "Nav.\nscore"], rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, float(np.nanmax(vals)) * 1.15))
    ax.set_title("Control analysis")
    _despine(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ------------------------- training/control analysis -------------------------

def save_training_control_analysis(history, metrics, out_dir):
    if history is None:
        return None

    set_nature_style_full(8)
    train_dir = os.path.join(out_dir, "nature_final_story", "training_control_analysis")
    ensure_dir(train_dir)
    df = pd.DataFrame(history).copy()
    if len(df) == 0:
        return None
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    if "val_event_f1" not in df.columns:
        df["val_event_f1"] = 0.0
    if "val_pseudo_action_agreement" not in df.columns:
        df["val_pseudo_action_agreement"] = 0.0
    if "train_loss" not in df.columns:
        df["train_loss"] = np.nan
    if "val_loss" not in df.columns:
        df["val_loss"] = np.nan

    df["perception_control_score"] = 0.50 * df["val_event_f1"].fillna(0) + 0.50 * df["val_pseudo_action_agreement"].fillna(0)
    df["control_consistency_index_proxy"] = (
        0.65 * df["perception_control_score"]
        + 0.20 * np.clip(metrics.get("risk_brake_alignment_plot", 0.0), 0, 1)
        + 0.15 * metrics.get("command_stability_plot", 0.0)
    )

    csv_path = os.path.join(train_dir, "training_control_analysis.csv")
    _save_dataframe(df, csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    axes[0].plot(df["epoch"], df["train_loss"], lw=1.0, color=NATURE_COLORS["gray"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], lw=1.0, color=NATURE_COLORS["blue"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(df["epoch"], df["val_event_f1"], lw=1.0, color=NATURE_COLORS["green"], label="event F1")
    axes[1].plot(df["epoch"], df["val_pseudo_action_agreement"], lw=1.0, color=NATURE_COLORS["orange"], label="pseudo-label acc.")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend(frameon=False)

    axes[2].plot(df["epoch"], df["control_consistency_index_proxy"], lw=1.2, color=NATURE_COLORS["purple"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Control consistency index")
    axes[2].set_title("Control-oriented training")

    for ax in axes:
        _despine(ax)
    fig.tight_layout(pad=0.4)
    png_path = os.path.join(train_dir, "training_control_analysis.png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training-control analysis:\n{png_path}")
    return png_path


# ------------------------- final composite figure -------------------------

def _save_final_panel_data(representative, command_df, metrics, out_dir, scene_elements_df, layer_df, network_node_df, network_edge_df, system_node_df, system_edge_df):
    final_dir = os.path.join(out_dir, "nature_final_story", "final_composite")
    panel_dir = os.path.join(final_dir, "panel_csv")
    ensure_dir(panel_dir)

    # a, b, c schematic data
    a_path = _save_dataframe(scene_elements_df, os.path.join(panel_dir, "panel_a_application_scene_elements.csv"))
    b_layer_path = _save_dataframe(layer_df, os.path.join(panel_dir, "panel_b_network_layer_table.csv"))
    b_node_path = _save_dataframe(network_node_df, os.path.join(panel_dir, "panel_b_network_nodes.csv"))
    b_edge_path = _save_dataframe(network_edge_df, os.path.join(panel_dir, "panel_b_network_edges.csv"))
    c_node_path = _save_dataframe(system_node_df, os.path.join(panel_dir, "panel_c_closed_loop_nodes.csv"))
    c_edge_path = _save_dataframe(system_edge_df, os.path.join(panel_dir, "panel_c_closed_loop_edges.csv"))

    # d feature maps, raw matrices in long table.
    x_np = representative["x_np"]
    risk = representative["risk"]
    feature_maps = {
        "night_frame": x_np[0],
        "etn_event_energy": x_np[2] + x_np[3],
        "motion_trace": x_np[4],
        "risk_map": risk,
    }
    long_dfs = [_long_matrix_dataframe(mat, value_name="value", feature_name=name) for name, mat in feature_maps.items()]
    d_df = pd.concat(long_dfs, ignore_index=True)
    d_path = _save_dataframe(d_df, os.path.join(panel_dir, "panel_d_feature_maps_long_pixels.csv"))

    # e/f/g/h data are derived from command dataframe and metrics.
    e_cols = ["frame", "time_s", "pred_action_id_raw", "pred_action_name_raw", "pred_action_id", "pred_action_name"]
    for name in ACTION_NAMES:
        e_cols += [f"p_raw_{name}", f"p_smooth_{name}"]
    e_path = _save_dataframe(command_df[e_cols].copy(), os.path.join(panel_dir, "panel_e_action_probabilities.csv"))

    f_cols = [
        "frame", "time_s", "risk_left_raw", "risk_center_raw", "risk_right_raw",
        "risk_left_smooth", "risk_center_smooth", "risk_right_smooth",
        "forward_velocity_cmd", "collective_thrust_cmd", "brake_cmd", "yaw_rate_cmd_deg_s",
        "pred_action_id", "pred_action_name"
    ]
    f_path = _save_dataframe(command_df[f_cols].copy(), os.path.join(panel_dir, "panel_f_command_modulation.csv"))

    g_cols = [
        "frame", "time_s", "traj_x_m", "traj_y_m", "traj_z_m", "heading_rad",
        "pred_action_id", "pred_action_name", "risk_center_smooth",
        "forward_velocity_cmd", "collective_thrust_cmd", "brake_cmd", "yaw_rate_cmd_deg_s"
    ]
    g_path = _save_dataframe(command_df[g_cols].copy(), os.path.join(panel_dir, "panel_g_trajectory_2d_3d.csv"))

    h_df = pd.DataFrame([
        ["Event F1", metrics["mean_event_f1"], metrics["mean_event_f1"], "Mean event-map F1 over plotted frames"],
        ["Risk-brake alignment", metrics["risk_brake_correlation_raw"], metrics["risk_brake_alignment_plot"], "Correlation between centre risk and brake command; negative values clipped only for plotting"],
        ["Turn-risk alignment", metrics["left_right_risk_yaw_correlation_raw"], metrics["turn_alignment_plot"], "Absolute correlation between left-right risk imbalance and yaw-rate command"],
        ["Command stability", 1.0 - metrics["action_switch_rate"], metrics["command_stability_plot"], "One minus the smoothed action switching rate"],
        ["Control consistency index", metrics["control_consistency_index"], metrics["control_consistency_index"], "Illustrative software-defined control consistency index; not flight success"],
    ], columns=["metric", "raw_value", "plot_value", "definition"])
    h_path = _save_dataframe(h_df, os.path.join(panel_dir, "panel_h_control_metrics.csv"))

    manifest_rows = [
        ["a", "Application scenario", a_path],
        ["b", "Network architecture layers", b_layer_path],
        ["b", "Network architecture nodes", b_node_path],
        ["b", "Network architecture edges", b_edge_path],
        ["c", "Closed-loop nodes", c_node_path],
        ["c", "Closed-loop edges", c_edge_path],
        ["d", "Representative feature maps", d_path],
        ["e", "Action probability heatmap", e_path],
        ["f", "Risk and command modulation", f_path],
        ["g", "2D and 3D trajectory", g_path],
        ["h", "Control analysis metrics", h_path],
    ]
    manifest_path = _save_dataframe(pd.DataFrame(manifest_rows, columns=["panel", "content", "csv_path"]), os.path.join(panel_dir, "final_composite_panel_manifest.csv"))
    return panel_dir, manifest_path


def _draw_feature_maps_panel(fig, parent_spec, representative):
    x_np = representative["x_np"]
    risk = representative["risk"]
    etn_energy = _safe01(x_np[2] + x_np[3])
    maps = [
        (x_np[0], "Night frame"),
        (etn_energy, "ETN event"),
        (x_np[4], "Motion trace"),
        (risk, "Risk field"),
    ]
    sub = parent_spec.subgridspec(2, 2, wspace=0.05, hspace=0.20)
    axes = []
    for i, (mat, title) in enumerate(maps):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        ax.imshow(mat, cmap="viridis", interpolation="nearest")
        ax.set_title(title, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        axes.append(ax)
    return axes


def _save_panel_d_feature_maps_png(representative, out_png):
    ensure_dir(os.path.dirname(out_png))
    x_np = representative["x_np"]
    risk = representative["risk"]
    maps = [
        (x_np[0], "Night frame"),
        (_safe01(x_np[2] + x_np[3]), "ETN event"),
        (x_np[4], "Motion trace"),
        (risk, "Risk field"),
    ]
    set_nature_style_full(8)
    fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.0))
    for ax, (mat, title) in zip(axes.reshape(-1), maps):
        ax.imshow(mat, cmap="viridis", interpolation="nearest")
        ax.set_title(title, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.06, hspace=0.24)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    return out_png


def _trim_white_pil(img, threshold=248, pad=12):
    from PIL import Image, ImageChops, ImageOps
    if img.mode != "RGB":
        img = img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L")
    # Make a binary mask of non-white content.
    bbox = diff.point(lambda p: 255 if p < threshold else 255).getbbox()
    # The line above intentionally keeps a valid bbox for dark panels. Use a stricter bbox below.
    import numpy as _np
    arr = _np.asarray(img)
    mask = _np.any(arr < threshold, axis=2)
    if mask.any():
        ys, xs = _np.where(mask)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(img.size[0], int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(img.size[1], int(ys.max()) + pad)
        if x1 > x0 and y1 > y0:
            img = img.crop((x0, y0, x1, y1))
    return img


def _fit_panel_to_cell(img_path, cell_w, cell_h, pad=48):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img = _trim_white_pil(img, threshold=248, pad=14)
    max_w = max(10, cell_w - 2 * pad)
    max_h = max(10, cell_h - 2 * pad)
    scale = min(max_w / img.size[0], max_h / img.size[1])
    new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
    x = (cell_w - new_size[0]) // 2
    y = (cell_h - new_size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def _make_final_montage(panel_items, out_png):
    from PIL import Image, ImageDraw, ImageFont
    cell_w, cell_h = 1550, 930
    margin_x, margin_y = 85, 75
    label_pad = 18
    n_cols, n_rows = 2, 4
    W = n_cols * cell_w + (n_cols + 1) * margin_x
    H = n_rows * cell_h + (n_rows + 1) * margin_y
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
    except Exception:
        label_font = ImageFont.load_default()

    for idx, item in enumerate(panel_items):
        row = idx // n_cols
        col = idx % n_cols
        x0 = margin_x + col * (cell_w + margin_x)
        y0 = margin_y + row * (cell_h + margin_y)
        panel_img = _fit_panel_to_cell(item["png"], cell_w, cell_h, pad=48)
        canvas.paste(panel_img, (x0, y0))
        draw.text((x0 + label_pad, y0 + label_pad), item["label"], fill=(34, 34, 34), font=label_font)

    ensure_dir(os.path.dirname(out_png))
    canvas.save(out_png, dpi=(600, 600))
    return out_png


def save_nature_final_composite_figure(
    representative,
    command_df,
    metrics,
    out_dir,
    scene_elements_df,
    layer_df,
    network_node_df,
    network_edge_df,
    system_node_df,
    system_edge_df
):
    final_dir = os.path.join(out_dir, "nature_final_story", "final_composite")
    panel_png_dir = os.path.join(final_dir, "panel_png")
    ensure_dir(panel_png_dir)

    panel_csv_dir, manifest_path = _save_final_panel_data(
        representative=representative,
        command_df=command_df,
        metrics=metrics,
        out_dir=out_dir,
        scene_elements_df=scene_elements_df,
        layer_df=layer_df,
        network_node_df=network_node_df,
        network_edge_df=network_edge_df,
        system_node_df=system_node_df,
        system_edge_df=system_edge_df,
    )

    # Existing clean standalone panels.
    panel_a_png = os.path.join(out_dir, "nature_final_story", "drone_application_scene", "drone_application_scene.png")
    panel_b_png = os.path.join(out_dir, "nature_final_story", "network_architecture", "network_architecture.png")
    panel_c_png = os.path.join(out_dir, "nature_final_story", "closed_loop_system", "closed_loop_system_diagram.png")
    panel_d_png = _save_panel_d_feature_maps_png(representative, os.path.join(panel_png_dir, "panel_d_feature_maps.png"))
    panel_e_png = os.path.join(out_dir, "nature_final_story", "drone_control_outputs", "panel_e_action_probability_heatmap.png")
    panel_f_png = os.path.join(out_dir, "nature_final_story", "drone_control_outputs", "drone_command_variation.png")
    panel_g_png = os.path.join(out_dir, "nature_final_story", "drone_control_outputs", "drone_motion_trace_2d.png")
    panel_h_png = os.path.join(out_dir, "nature_final_story", "drone_control_outputs", "drone_control_metrics.png")

    panel_items = [
        {"label": "a", "title": "Application scenario", "png": panel_a_png},
        {"label": "b", "title": "Network architecture", "png": panel_b_png},
        {"label": "c", "title": "Closed-loop pathway", "png": panel_c_png},
        {"label": "d", "title": "Feature maps", "png": panel_d_png},
        {"label": "e", "title": "Action probabilities", "png": panel_e_png},
        {"label": "f", "title": "Command modulation", "png": panel_f_png},
        {"label": "g", "title": "Trajectory", "png": panel_g_png},
        {"label": "h", "title": "Control metrics", "png": panel_h_png},
    ]

    png_manifest = pd.DataFrame(panel_items)
    png_manifest["csv_folder"] = panel_csv_dir
    _save_dataframe(png_manifest, os.path.join(panel_png_dir, "final_composite_panel_png_manifest.csv"))

    png_path = os.path.join(final_dir, "nature_final_full_story_figure.png")
    _make_final_montage(panel_items, png_path)

    print(f"Saved revised Nature final full-story montage:\n{png_path}")
    print(f"Saved final panel CSV manifest:\n{manifest_path}")
    return png_path


def save_nature_full_story_outputs(model, dataset, device, out_dir, history=None, num_examples=6, max_frames=450, threshold=0.50):
    story_dir = os.path.join(out_dir, "nature_final_story")
    ensure_dir(story_dir)

    network_png, layer_df, network_node_df, network_edge_df = save_network_architecture_diagram(model=model, out_dir=out_dir)
    scene_png, scene_elements_df = save_drone_application_scene(out_dir=out_dir)
    system_png, system_node_df, system_edge_df = save_closed_loop_system_diagram(out_dir=out_dir)

    representative = save_all_2d_3d_feature_outputs(
        model=model,
        dataset=dataset,
        device=device,
        out_dir=out_dir,
        num_examples=num_examples,
        threshold=threshold
    )

    command_df, metrics = save_drone_command_and_trajectory_outputs(
        model=model,
        dataset=dataset,
        device=device,
        out_dir=out_dir,
        max_frames=max_frames,
        threshold=threshold
    )

    save_training_control_analysis(history=history, metrics=metrics, out_dir=out_dir)

    if representative is not None:
        save_nature_final_composite_figure(
            representative=representative,
            command_df=command_df,
            metrics=metrics,
            out_dir=out_dir,
            scene_elements_df=scene_elements_df,
            layer_df=layer_df,
            network_node_df=network_node_df,
            network_edge_df=network_edge_df,
            system_node_df=system_node_df,
            system_edge_df=system_edge_df
        )

    print("\nAll revised Nature full-story outputs saved in:")
    print(story_dir)
    return story_dir


# ============================================================
# Public-release dataset split and run provenance
# ============================================================

def concatenate_sequence_caches(paths, combined_path, args, etn_params):
    """Build caches with independent temporal states; combine without mixing sequences."""
    ensure_dir(os.path.dirname(combined_path))
    xs, ys, acts, metas = [], [], [], []
    for sequence_i, path in enumerate(paths):
        label = f"sequence_{sequence_i:02d}_{os.path.splitext(os.path.basename(path))[0]}"
        seq_path = combined_path.replace(".npz", f"_{label}.npz")
        build_cache(path, seq_path, height=args.height, width=args.width,
                    max_samples=args.max_samples, frame_step=args.frame_step,
                    max_events=args.max_events, rebuild=args.rebuild_cache,
                    etn_params=etn_params, camera_fps=args.camera_fps)
        with np.load(seq_path) as data:
            xs.append(data["X"]); ys.append(data["Y_event"]); acts.append(data["Y_action"])
        meta = pd.read_csv(seq_path.replace(".npz", "_meta.csv"))
        meta["source_sequence_id"] = label
        metas.append(meta)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    A = np.concatenate(acts, axis=0)
    combined_meta = pd.concat(metas, ignore_index=True)
    combined_meta["cache_index"] = np.arange(len(combined_meta))
    # Regenerate action columns as fixed-rule pseudo-labels
    combined_meta["fixed_rule_pseudo_action_id"] = A
    combined_meta["fixed_rule_pseudo_action_name"] = [ACTION_NAMES[int(v)] for v in A]
    tmp = combined_path + ".tmp.npz"
    np.savez_compressed(tmp, X=X, Y_event=Y, Y_action=A)
    os.replace(tmp, combined_path)
    combined_meta.to_csv(combined_path.replace(".npz", "_meta.csv"), index=False)
    return combined_path


def make_leakage_aware_split(metadata, train_fraction=0.8, purge_frames=30,
                             val_sequence=None):
    """Independent-sequence holdout preferred; otherwise purged chronological split.

    This never claims that one sequence split validates new environments.
    """
    if not 0.1 < train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.1 and 0.9")
    seqs = list(dict.fromkeys(metadata["source_sequence_id"].astype(str).tolist()))
    if len(seqs) >= 2:
        chosen = seqs[-1] if val_sequence is None else str(val_sequence)
        if chosen not in seqs:
            raise ValueError(f"Validation sequence {chosen} not present in {seqs}")
        train_idx = metadata.index[metadata["source_sequence_id"].astype(str) != chosen].to_numpy()
        val_idx = metadata.index[metadata["source_sequence_id"].astype(str) == chosen].to_numpy()
        strategy = "independent_sequence_holdout"
    else:
        ordered = metadata.sort_values("frame_idx", kind="stable")
        cutoff = int(len(ordered) * train_fraction)
        train = ordered.iloc[:cutoff]
        val = ordered.iloc[cutoff:]
        if not len(train) or not len(val):
            raise ValueError("Insufficient samples to create chronological split")
        left_boundary = int(train["frame_idx"].max())
        right_boundary = int(val["frame_idx"].min())
        # Purge around boundary in units of SOURCE VIDEO FRAMES, not shuffled cache indices
        train_idx = train.index[train["frame_idx"] <= right_boundary - purge_frames - 1].to_numpy()
        val_idx = val.index[val["frame_idx"] >= left_boundary + purge_frames + 1].to_numpy()
        strategy = "chronological_single_sequence_with_embargo_not_independent_test"
    if len(train_idx) < 1 or len(val_idx) < 1:
        raise ValueError("Empty split after temporal embargo. Add sequences/data or adjust --purge_frames with justification")
    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise AssertionError("Train-validation overlap")
    return np.asarray(sorted(train_idx), dtype=int), np.asarray(sorted(val_idx), dtype=int), strategy


def write_release_manifest(args, paths, cache_path, strategy, train_idx, val_idx):
    sim = load_json_config(args.simulation_config)
    fit = load_json_config(args.device_fit_config)
    manifest = {
        "title": "Experimentally Parameterized ETN Model for Dataset-Based Neuromorphic Navigation Simulation",
        "run_type": "dataset_based_software_simulation_not_physical_drone_experiment",
        "parameter_layers": {
            "input_parameter_points": {"path": os.path.basename(args.figure_points_csv),
                                       "sha256": file_sha256(args.figure_points_csv),
                                       "source_types": sorted(FIGURE_POINTS_DF["source_type"].unique().tolist())},
            "device_fit": {"config": fit, "sha256": file_sha256(args.device_fit_config)},
            "software_simulation": {"config": sim, "sha256": file_sha256(args.simulation_config)},
        },
        "hdf5_sources": [{"basename": os.path.basename(p), "size_bytes": os.path.getsize(p)} for p in paths],
        "validation_split": {"strategy": strategy, "train_count": len(train_idx),
                             "val_count": len(val_idx), "purge_frames": args.purge_frames,
                             "note": "Single-sequence split cannot establish cross-scene generalization."},
        "pseudo_labels": "Fixed hand-authored software risk rules, not flight/action ground truth",
        "optical_input": "Assumed grayscale-to-NIR-power proxy; not optical measurement",
        "navigation_trajectory": "Illustrative command-driven integration; not measured vehicle motion",
        "framework_versions": {"python": __import__("sys").version.split()[0],
                               "torch": torch.__version__, "numpy": np.__version__,
                               "pandas": pd.__version__, "h5py": h5py.__version__,
                               "opencv": cv2.__version__},
        "seed": args.seed,
        "command_args": {k: ("<local-path-omitted>" if k in (
            "hdf5", "result_dir", "figure_points_csv", "device_fit_config", "simulation_config")
            else v) for k, v in vars(args).items()}
    }
    with open(os.path.join(args.result_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ============================================================
# 11. Main
# ============================================================

def main():
    # Read config locations first, so external files control argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--device_fit_config", default=DEFAULT_DEVICE_FIT_CONFIG)
    pre.add_argument("--simulation_config", default=DEFAULT_SIMULATION_CONFIG)
    pre.add_argument("--figure_points_csv", default=DEFAULT_FIGURE_POINTS)
    early, _ = pre.parse_known_args()
    simulation_defaults = load_json_config(early.simulation_config)
    param_defaults = default_etn_params(early.device_fit_config, early.simulation_config)
    parser = argparse.ArgumentParser(parents=[pre])

    parser.add_argument("--hdf5", nargs="+", default=None,
                        help="One or more separate MVSEC HDF5 sequences; holdout last by default")
    parser.add_argument("--val_sequence", default=None,
                        help="Named source sequence to hold out (when multiple HDF5 supplied)")
    parser.add_argument("--train_fraction", type=float, default=simulation_defaults["train_val_ratio"])
    parser.add_argument("--purge_frames", type=int, default=simulation_defaults["purge_frames"])

    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)

    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=176)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_events", type=int, default=200000)
    parser.add_argument("--camera_fps", type=float, default=simulation_defaults["camera_fps"],
                        help="fallback only when MVSEC image timestamps are unavailable")
    for k, v in param_defaults.items():
        parser.add_argument("--"+k, type=float, default=v)
    parser.add_argument("--calibration_only", action="store_true",
                        help="export figure-digitized-point interpolation and illustrations; no MVSEC required")

    parser.add_argument("--epochs", type=int, default=simulation_defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=simulation_defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=simulation_defaults["lr"])
    parser.add_argument("--lambda_action", type=float, default=simulation_defaults["lambda_action"])
    parser.add_argument("--lambda_focal", type=float, default=simulation_defaults["lambda_focal"])
    parser.add_argument("--base_channels", type=int, default=simulation_defaults["base_channels"])

    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_visuals", action="store_true")
    parser.add_argument("--vis_examples", type=int, default=8)
    parser.add_argument("--video_frames", type=int, default=200)
    parser.add_argument("--video_fps", type=int, default=12)
    parser.add_argument("--inference_save_max", type=int, default=200)
    parser.add_argument("--three_d_frames", type=int, default=120)

    args = parser.parse_args()

    set_seed(args.seed)
    set_figure_points(args.figure_points_csv)
    ensure_dir(args.result_dir)
    etn_params = {k:getattr(args,k) for k in param_defaults}
    save_experimental_etn_calibration(os.path.join(args.result_dir,"etn_device_calibration"),
                                      etn_params, figure_points_csv=args.figure_points_csv)
    if args.calibration_only:
        print("Parameter-lookup export completed; MVSEC dataset not required.")
        return

    device = torch.device("cpu")
    print(f"Using device: {device}")

    hdf5_paths = args.hdf5
    if hdf5_paths is None:
        found = find_mvsec_hdf5()
        hdf5_paths = [found] if found else []
    hdf5_paths = [os.path.abspath(p) for p in hdf5_paths]
    if not hdf5_paths or not all(os.path.isfile(p) for p in hdf5_paths):
        raise FileNotFoundError("MVSEC input not found. Supply --hdf5 path1.hdf5 [path2.hdf5 ...]. For model calibration without MVSEC, run --calibration_only. The embedded example parameters are not a replacement for the MVSEC dataset.")
    if len(hdf5_paths) != len(set(hdf5_paths)):
        raise ValueError("Duplicate source sequence: the same HDF5 cannot be in both splits")
    input_signature = [{"path":p,"size":os.path.getsize(p),"mtime_ns":os.stat(p).st_mtime_ns} for p in hdf5_paths]
    config_signature = hashlib.sha256(json.dumps({
        "etn":etn_params,"version":ETN_CALIBRATION_VERSION,
        "figure_points_sha256":file_sha256(args.figure_points_csv),
        "device_fit_sha256":file_sha256(args.device_fit_config),
        "simulation_sha256":file_sha256(args.simulation_config),
        "camera_fps":args.camera_fps,
        "max_events":args.max_events,
        "inputs":input_signature,
    }, sort_keys=True).encode()).hexdigest()[:12]
    cache_name = (f"multi_seq_expcal_{config_signature}_etn_cache_"
                  f"{args.width}x{args.height}_N{args.max_samples}_S{args.frame_step}.npz")
    cache_path = os.path.join(args.result_dir, cache_name)
    if args.rebuild_cache or not cache_is_valid(cache_path) or not os.path.isfile(cache_path.replace(".npz", "_meta.csv")):
        concatenate_sequence_caches(hdf5_paths, cache_path, args, etn_params)
    metadata = pd.read_csv(cache_path.replace(".npz", "_meta.csv"))
    # The combined cache's row order exactly defines cache_index.
    if not np.array_equal(metadata["cache_index"].to_numpy(), np.arange(len(metadata))):
        raise AssertionError("Cache index and metadata mismatch")
    train_idx, val_idx, split_strategy = make_leakage_aware_split(
        metadata, train_fraction=args.train_fraction, purge_frames=args.purge_frames,
        val_sequence=args.val_sequence)
    print("Validation strategy:", split_strategy, "| train", len(train_idx), "| val", len(val_idx))
    save_train_val_indices(args.result_dir, train_idx, val_idx)
    write_release_manifest(args, hdf5_paths, cache_path, split_strategy, train_idx, val_idx)

    train_dataset = ETNMVSECDataset(cache_path, train_idx)
    val_dataset = ETNMVSECDataset(cache_path, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    pos_weight = compute_pos_weight(train_dataset).to(device)
    print(f"\nBCE pos_weight: {pos_weight.cpu().numpy()}")

    event_bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.view(1, 2, 1, 1)
    )

    action_weight = compute_action_class_weight(train_dataset).to(device)
    action_loss_fn = nn.CrossEntropyLoss(weight=action_weight)

    model = ETNMotionNet(
        in_ch=7,
        base=args.base_channels,
        num_actions=len(ACTION_NAMES)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.result_dir, "best_etn_motionnet_mvsec_stable.pth")

    history = []

    print("\nStart training...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"lambda_action: {args.lambda_action}")
    print(f"lambda_focal:  {args.lambda_focal}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            event_bce_loss_fn=event_bce_loss_fn,
            action_loss_fn=action_loss_fn,
            device=device,
            lambda_action=args.lambda_action,
            lambda_focal=args.lambda_focal,
            grad_clip=1.0
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            event_bce_loss_fn=event_bce_loss_fn,
            action_loss_fn=action_loss_fn,
            device=device,
            lambda_action=args.lambda_action,
            lambda_focal=args.lambda_focal
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_event_loss": train_metrics["event_loss"],
            "train_action_loss": train_metrics["action_loss"],
            "train_pseudo_action_agreement": train_metrics["pseudo_action_agreement"],
            "val_loss": val_metrics["loss"],
            "val_event_loss": val_metrics["event_loss"],
            "val_action_loss": val_metrics["action_loss"],
            "val_pseudo_action_agreement": val_metrics["pseudo_action_agreement"],
            "val_event_precision": val_metrics["event_precision"],
            "val_event_recall": val_metrics["event_recall"],
            "val_event_f1": val_metrics["event_f1"],
            "lr": optimizer.param_groups[0]["lr"]
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val loss {row['val_loss']:.4f} | "
            f"val F1 {row['val_event_f1']:.4f} | "
            f"val pseudo-label acc {row['val_pseudo_action_agreement']:.4f} | "
            f"lr {row['lr']:.2e}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "action_names": ACTION_NAMES,
                    "best_val_loss": best_val_loss,
                },
                best_path
            )
            print(f"Saved best model: {best_path}")

    plot_training_curves(history, args.result_dir)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_action_confusion_matrix(
        model=model,
        dataset=val_dataset,
        device=device,
        out_dir=args.result_dir
    )

    save_inference_data(
        model=model,
        dataset=val_dataset,
        device=device,
        out_dir=args.result_dir,
        max_save=args.inference_save_max,
        threshold=0.50
    )

    if not args.no_visuals:
        save_static_feature_maps(
            model=model,
            dataset=val_dataset,
            device=device,
            out_dir=args.result_dir,
            num_examples=args.vis_examples,
            threshold=0.50
        )

        save_feature_video(
            model=model,
            dataset=val_dataset,
            device=device,
            out_dir=args.result_dir,
            video_frames=args.video_frames,
            fps=args.video_fps,
            threshold=0.50
        )

        save_3d_feature_plots(
            dataset=val_dataset,
            out_dir=args.result_dir,
            sample_count=args.three_d_frames,
            selected_examples=4
        )

        save_nature_full_story_outputs(
            model=model,
            dataset=val_dataset,
            device=device,
            out_dir=args.result_dir,
            history=history,
            num_examples=args.vis_examples,
            max_frames=min(args.video_frames, 450),
            threshold=0.50
        )

    print("\nTraining and inference finished.")
    print(f"Best model:\n{best_path}")
    print(f"Results folder:\n{args.result_dir}")
    print("\nSaved outputs include:")
    print("0. etn_device_calibration/ digitized input points and interpolation diagnostics")
    print("1. training_metrics.csv")
    print("2. loss/action/event metric curves")
    print("3. train_val_indices.csv")
    print("4. action_confusion_matrix.png (agreement with simulation-rule pseudo-labels)")
    print("5. action_confusion_matrix_normalized.png")
    print("6. inference_data/inference_summary.csv")
    print("7. inference_data/inference_maps_subset.npz")
    print("8. static_feature_maps/")
    print("9. feature_extraction_video/etn_feature_extraction_video.mp4")
    print("10. three_dimensional_feature_plots/")
    print("11. nature_final_story/network_architecture/network_architecture.png + CSV")
    print("12. nature_final_story/drone_application_scene/drone_application_scene.png + CSV")
    print("13. nature_final_story/closed_loop_system/closed_loop_system_diagram.png + CSV")
    print("14. nature_final_story/feature_2d_3d_outputs/ all 2D and 3D CSV + PNG")
    print("15. nature_final_story/drone_control_outputs/ command, 2D trace, 3D trace CSV + PNG")
    print("16. nature_final_story/training_control_analysis/ training-control CSV + PNG")
    print("17. nature_final_story/final_composite/nature_final_full_story_figure.png")


if __name__ == "__main__":
    main()

