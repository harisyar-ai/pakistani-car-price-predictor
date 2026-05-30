# ─────────────────────────────────────────────────────────────
#  PakWheels Used Car Price Predictor — 2026 Research Study
#  Author: Muhammad Haris Afridi
#  Dept. of Computer Science, Khyber Pakhtunkhwa
# ─────────────────────────────────────────────────────────────

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os
import re
from html import escape
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistani Cars Price Predictor | 2026 Study",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── VIEWPORT + SUPPRESS BALLOONS ─────────────────────────────
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<script>
window._streamlitBalloons = () => {};
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  GLOBAL STYLES  — Red & Black, Nova Books architecture
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --red:      #FF4B4B;
    --red2:     #FF7070;
    --red-dim:  #CC2E2E;
    --dark:     #0A0A0A;
    --dark2:    #111111;
    --dark3:    #1A1A1A;
    --dark4:    #222222;
    --border:   rgba(255,75,75,0.18);
    --border2:  rgba(255,75,75,0.35);
    --muted:    #888888;
    --white:    #F0F0F0;
    --dimwhite: #BBBBBB;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] {
    visibility: visible;
    background: transparent;
}
.stDeployButton { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── APP BACKGROUND ── */
.stApp {
    background-color: var(--dark);
    background-image:
        radial-gradient(ellipse at 15% 0%,  rgba(255,75,75,0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 100%, rgba(180,30,30,0.09) 0%, transparent 55%);
    font-family: 'Inter', sans-serif;
    color: var(--white);
}

.block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1300px !important;
    position: relative;
    z-index: 1;
}

/* ── TOP HEADER BAR ── */
.pw-header {
    background: linear-gradient(135deg, rgba(10,10,10,0.98) 0%, rgba(22,10,10,0.98) 100%);
    border-bottom: 1px solid var(--border2);
    padding: 24px 48px 20px;
    margin: -1rem -2rem 2.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.pw-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
}
.pw-logo {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--red);
    letter-spacing: 0.06em;
    text-shadow: 0 0 40px rgba(255,75,75,0.45);
    line-height: 1;
}
.pw-logo span { color: var(--white); font-weight: 400; }
.pw-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 5px;
}
.pw-badge {
    background: rgba(255,75,75,0.1);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 6px 16px;
    font-size: 0.72rem;
    color: var(--red2);
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── SECTION HEADINGS ── */
.section-heading {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.75rem;
    color: var(--white);
    font-weight: 700;
    margin: 2rem 0 0.3rem;
    border-left: 3px solid var(--red);
    padding-left: 16px;
    letter-spacing: 0.04em;
}
.section-divider {
    width: 60px; height: 1px;
    background: var(--red);
    margin: 0 0 1.8rem 19px;
    opacity: 0.5;
}

/* ── INPUT FORM PANEL ── */
.form-panel {
    background: linear-gradient(160deg, rgba(22,10,10,0.9) 0%, rgba(15,10,10,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 1.5rem;
}
.form-section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    color: var(--red);
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── PREDICTION RESULT BOX ── */
.result-box {
    background: linear-gradient(135deg, rgba(255,75,75,0.08) 0%, rgba(15,10,10,0.96) 100%);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 36px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin: 1.5rem 0;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
}
.result-label {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-price {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--red);
    letter-spacing: 0.04em;
    text-shadow: 0 0 40px rgba(255,75,75,0.4);
    line-height: 1.1;
}
.result-range {
    font-size: 1rem;
    color: var(--dimwhite);
    margin-top: 8px;
    letter-spacing: 0.05em;
}
.result-urdu {
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 6px;
    direction: rtl;
}
.confidence-badge {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 18px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.conf-high   { background: rgba(50,200,100,0.12); color: #50C878; border: 1px solid rgba(50,200,100,0.3); }
.conf-good   { background: rgba(255,185,50,0.12); color: #FFB932; border: 1px solid rgba(255,185,50,0.3); }
.conf-mod    { background: rgba(255,75,75,0.12);  color: var(--red2); border: 1px solid var(--border2); }

/* ── CAR LISTING CARD (hover) ── */
.car-card-wrap { text-decoration: none !important; display: block; }
.car-card {
    background: linear-gradient(160deg, rgba(22,12,12,0.92) 0%, rgba(14,10,10,0.96) 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    cursor: pointer;
    position: relative;
}
.car-card::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(160deg, rgba(255,75,75,0.06), transparent);
    opacity: 0; transition: opacity 0.35s;
    pointer-events: none;
    z-index: 1;
}
.car-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: var(--border2);
    box-shadow: 0 20px 50px rgba(0,0,0,0.7), 0 0 30px rgba(255,75,75,0.12);
}
.car-card:hover::before { opacity: 1; }
.car-card-img {
    width: 100%;
    height: 160px;
    object-fit: cover;
    display: block;
    transition: transform 0.4s ease;
}
.car-card:hover .car-card-img { transform: scale(1.04); }
.car-card-img-wrap { overflow: hidden; height: 160px; }
.car-card-body { padding: 12px 14px 16px; }
.car-card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 4px;
    letter-spacing: 0.03em;
}
.car-card-meta {
    font-size: 0.78rem;
    color: var(--muted);
    display: flex;
    gap: 10px;
    align-items: center;
}
.car-card-meta span { color: var(--dimwhite); }
.car-card-link {
    display: block;
    margin-top: 10px;
    font-size: 0.72rem;
    color: var(--red);
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,75,75,0.25);
    width: fit-content;
    transition: color 0.2s, border-color 0.2s;
}
.car-card:hover .car-card-link { color: var(--red2); border-color: rgba(255,112,112,0.5); }
.car-card-placeholder {
    width: 100%; height: 160px;
    background: linear-gradient(135deg, var(--dark3), var(--dark4));
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem;
}

/* ── STREAMLIT WIDGETS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(16,10,10,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
}
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: var(--muted) !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--border2) !important;
    box-shadow: 0 0 0 1px rgba(255,75,75,0.2) !important;
}

/* ── BASEWEB DROPDOWN ── */
[data-baseweb="select"] * { border: none !important; outline: none !important; box-shadow: none !important; }
[data-baseweb="select"] > div {
    background: rgba(16,10,10,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] * { background-color: #140A0A !important; border: none !important; outline: none !important; box-shadow: none !important; }
[data-baseweb="popover"] > div { border: 1px solid var(--border2) !important; border-radius: 4px !important; }
[data-baseweb="menu"], [data-baseweb="menu"] * { background-color: #140A0A !important; border: none !important; }
[data-baseweb="option"] {
    background-color: #140A0A !important;
    color: var(--dimwhite) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    border: none !important;
}
[data-baseweb="option"]:hover,
[data-baseweb="option"]:focus { background-color: rgba(255,75,75,0.12) !important; color: var(--red2) !important; }
li[aria-selected="true"],
[role="option"][aria-selected="true"] { background-color: rgba(255,75,75,0.18) !important; color: var(--red2) !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--red) 0%, var(--red-dim) 100%) !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--red2) 0%, var(--red) 100%) !important;
    box-shadow: 0 6px 24px rgba(255,75,75,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,5,5,0.99) 0%, rgba(12,8,8,0.99) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    visibility: visible !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
    padding: 1.4rem 1.2rem !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--dimwhite) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 0.35rem !important;
}
[data-testid="stSidebar"] .stRadio [role="radio"] {
    background: rgba(255,75,75,0.05) !important;
    border: 1px solid rgba(255,75,75,0.12) !important;
    border-radius: 6px !important;
    padding: 0.55rem 0.7rem !important;
}
[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] {
    background: rgba(255,75,75,0.14) !important;
    border-color: var(--border2) !important;
}
[data-testid="stSidebar"] .stRadio [role="radio"] p {
    color: var(--dimwhite) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] p {
    color: var(--white) !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── INSIGHT IMAGE CARD ── */
.insight-card {
    background: linear-gradient(135deg, rgba(22,12,12,0.92), rgba(14,10,10,0.96));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 1.5rem;
}
.insight-caption {
    font-size: 0.83rem;
    color: var(--dimwhite);
    margin-top: 14px;
    line-height: 1.7;
    padding: 12px 16px;
    background: rgba(255,75,75,0.04);
    border-left: 2px solid var(--red);
    border-radius: 0 4px 4px 0;
}

/* ── ABOUT PAGE ── */
.about-card {
    background: linear-gradient(135deg, rgba(22,12,12,0.92), rgba(14,10,10,0.96));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 20px;
    height: 100%;
}
.about-card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    color: var(--red);
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.about-stat { display: flex; align-items: center; gap: 14px; padding: 11px 0; border-bottom: 1px solid rgba(255,75,75,0.07); }
.about-stat:last-child { border-bottom: none; }
.about-stat-icon { font-size: 1.2rem; width: 30px; text-align: center; }
.about-stat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; }
.about-stat-value { font-size: 0.88rem; color: var(--white); font-weight: 500; }

/* ── METRIC CHIPS ── */
.metric-chip {
    background: rgba(255,75,75,0.08);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-chip-val { font-family: 'Rajdhani', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--red); }
.metric-chip-lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.14em; margin-top: 2px; }

/* ── FOOTER ── */
.pw-footer {
    background: linear-gradient(135deg, rgba(8,5,5,0.99), rgba(14,8,8,0.99));
    border-top: 1px solid var(--border);
    padding: 36px 48px 24px;
    margin: 4rem -2rem -4rem;
    text-align: center;
}
.pw-footer::before {
    content: '— ✦ —';
    display: block;
    font-size: 0.9rem;
    color: var(--red);
    opacity: 0.4;
    margin-bottom: 16px;
    letter-spacing: 0.4em;
}
.pw-footer-logo { font-family: 'Rajdhani', sans-serif; font-size: 1.4rem; color: var(--red); font-weight: 700; margin-bottom: 5px; letter-spacing: 0.08em; }
.pw-footer-sub { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 18px; }
.pw-footer-copy { font-size: 0.7rem; color: rgba(136,136,136,0.45); letter-spacing: 0.1em; }

/* ── MISC ── */
hr { border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--red) !important; }
.stAlert {
    background: rgba(255,75,75,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--dimwhite) !important;
}
.stInfo { background: rgba(255,75,75,0.05) !important; border: 1px solid var(--border) !important; }

/* ── NUMBER INPUT ARROWS FIX ── */
.stNumberInput input { color: var(--white) !important; }

/* ── HIDE ORPHANED FORM-PANEL WRAPPERS ── */
.form-panel { display: contents !important; }

/* ── PAGINATION CARD ANIMATION ── */
@keyframes cardSlideIn {
    0%   { opacity: 0; transform: translateY(18px) scale(0.97); }
    60%  { opacity: 1; transform: translateY(-3px) scale(1.005); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
.cards-animated .car-card {
    animation: cardSlideIn 0.38s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}
.cards-animated .car-card:nth-child(1) { animation-delay: 0.00s; }
.cards-animated .car-card:nth-child(2) { animation-delay: 0.06s; }
.cards-animated .car-card:nth-child(3) { animation-delay: 0.12s; }
.cards-animated .car-card:nth-child(4) { animation-delay: 0.18s; }
.cards-animated .car-card:nth-child(5) { animation-delay: 0.24s; }

/* ── AUTHOR PAGE ── */
.author-hero {
    display: flex;
    align-items: center;
    gap: 36px;
    background: linear-gradient(135deg, rgba(22,10,10,0.92) 0%, rgba(14,10,10,0.96) 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 32px 36px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.author-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
}
.author-avatar {
    width: 200px; height: 200px;
    border-radius: 50%;
    border: 2px solid var(--border2);
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: 0 0 30px rgba(255,75,75,0.15);
}
.author-hero-info { flex: 1; }
.author-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 0.04em;
    line-height: 1.1;
    margin-bottom: 4px;
}
.author-role {
    font-size: 0.78rem;
    color: var(--red);
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.author-bio {
    font-size: 0.88rem;
    color: var(--dimwhite);
    line-height: 1.75;
    max-width: 600px;
}
.author-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.author-badge {
    background: rgba(255,75,75,0.08);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: var(--dimwhite);
    font-weight: 600;
    letter-spacing: 0.08em;
}
.author-socials {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 1.2rem;
    margin-bottom: 1.8rem;
}
.social-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 10px 20px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: rgba(22,10,10,0.9);
    color: var(--dimwhite);
    font-size: 0.82rem;
    font-weight: 600;
    text-decoration: none;
    letter-spacing: 0.06em;
    transition: all 0.25s;
}
.social-pill:hover {
    border-color: var(--border2);
    color: var(--red2);
    background: rgba(255,75,75,0.06);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255,75,75,0.12);
}
.social-pill i { font-size: 1rem; }
.social-pill.github i  { color: #aabbff; }
.social-pill.linkedin i { color: #48afff; }
.social-pill.email i   { color: #57d7a7; }
.social-pill.whatsapp i { color: #25d366; }
.social-pill.portfolio i { color: var(--red2); }
.author-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 1.6rem;
}
.author-info-card {
    background: linear-gradient(135deg, rgba(22,10,10,0.92), rgba(14,10,10,0.96));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 22px 24px;
}
.author-info-card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    color: var(--red);
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.author-info-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255,75,75,0.05);
}
.author-info-row:last-child { border-bottom: none; }
.author-info-icon { font-size: 1rem; width: 22px; text-align: center; flex-shrink: 0; margin-top: 1px; }
.author-info-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.author-info-value { font-size: 0.86rem; color: var(--white); font-weight: 500; }

/* ══════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE
   ══════════════════════════════════════════════════════════════ */

/* ── CARD GRID: override Streamlit's fixed columns with a wrapping grid ── */
/* On mobile, cards-animated children wrap into a 1-col or 2-col grid     */
.cards-animated [data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
}
.cards-animated [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
.cards-animated [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 280px !important;
    min-width: 260px !important;
    max-width: 100% !important;
}

@media (max-width: 768px) {

    /* ── GLOBAL PADDING ── */
    .block-container {
        padding: 0 0.75rem 3rem 0.75rem !important;
    }

    /* ── HEADER ── */
    .pw-header {
        padding: 16px 20px 14px !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 10px !important;
    }
    .pw-logo { font-size: 1.4rem !important; }
    .pw-tagline { font-size: 0.62rem !important; }
    .pw-badge { font-size: 0.65rem !important; padding: 4px 10px !important; }

    /* ── SECTION HEADINGS ── */
    .section-heading { font-size: 1.3rem !important; }

    /* ── STREAMLIT COLUMNS: force all multi-col layouts to stack ── */
    /* Targets the horizontal flex containers Streamlit generates    */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: min(100%, 300px) !important;
        flex: 1 1 300px !important;
    }

    /* ── FORM PANELS ── */
    .form-panel { padding: 16px 14px !important; }
    .form-section-label { font-size: 0.65rem !important; }

    /* ── RESULT BOX ── */
    .result-box { padding: 24px 18px !important; }
    .result-price { font-size: 2rem !important; }
    .result-range { font-size: 0.88rem !important; }

    /* ── CAR CARDS ── */
    .car-card { margin-bottom: 0 !important; }
    .car-card:hover {
        transform: none !important;
        box-shadow: none !important;
    }
    .car-card:active {
        transform: scale(0.98) !important;
        border-color: var(--border2) !important;
    }
    .car-card-img, .car-card-img-wrap { height: 130px !important; }
    .car-card-title { font-size: 0.92rem !important; }

    /* ── AUTHOR HERO ── */
    .author-hero {
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        padding: 24px 18px !important;
        gap: 20px !important;
    }
    .author-avatar {
        width: 120px !important;
        height: 120px !important;
    }
    .author-name { font-size: 1.4rem !important; }
    .author-bio { font-size: 0.82rem !important; max-width: 100% !important; }
    .author-badges { justify-content: center !important; }
    .author-socials { justify-content: center !important; }

    /* ── AUTHOR INFO GRID: 2-col → 1-col ── */
    .author-info-grid {
        grid-template-columns: 1fr !important;
    }

    /* ── ABOUT CARDS ── */
    .about-card { padding: 18px 16px !important; }

    /* ── INSIGHT CARD ── */
    .insight-card { padding: 14px !important; }

    /* ── FOOTER ── */
    .pw-footer {
        padding: 24px 20px 16px !important;
        margin: 3rem -0.75rem -3rem !important;
    }
    .pw-footer-logo { font-size: 1.1rem !important; }
    .pw-footer-sub { font-size: 0.62rem !important; }
    .pw-footer-copy { font-size: 0.6rem !important; }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1rem 0.8rem !important;
    }

    /* ── HIDE STREAMLIT SIDEBAR COLLAPSE BUTTON OVERLAP ── */
    [data-testid="collapsedControl"] {
        top: 12px !important;
    }
}

@media (max-width: 480px) {

    /* ── SINGLE COLUMN CARDS on very small screens ── */
    .cards-animated [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
    .cards-animated [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* ── HEADER ── */
    .pw-logo { font-size: 1.2rem !important; }
    .pw-header { padding: 12px 14px !important; }

    /* ── RESULT PRICE ── */
    .result-price { font-size: 1.75rem !important; }

    /* ── PREV/NEXT BUTTONS ── */
    .stButton > button {
        font-size: 0.75rem !important;
        padding: 0.55rem 0.8rem !important;
        letter-spacing: 0.08em !important;
    }

    /* ── SOCIAL PILLS ── */
    .social-pill {
        padding: 8px 14px !important;
        font-size: 0.75rem !important;
        gap: 6px !important;
    }

    /* ── METRIC CHIPS ── */
    .metric-chip-val { font-size: 1.4rem !important; }
}

</style>
""", unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CUSTOM SKLEARN CLASSES  (must be defined before joblib.load)
# ══════════════════════════════════════════════════════════════

NUMERIC_COLS       = ['car_age', 'Mileage', 'Engine_CC_Clean']
TARGET_ENCODE_COLS = ['brand', 'model_s4', 'brand_model_generation']
OHE_COLS           = ['generation', 'Fuel_Type', 'Transmission', 'brand_origin', 'city_tier',
                       'trim_tier_s4', 'trim_grade_s4']
BINARY_COLS        = ['is_electric']

class FeaturePrep(BaseEstimator, TransformerMixin):
    def __init__(self, rare_trim_threshold=30, rare_model_threshold=30,
                 rare_generation_threshold=30, rare_model_generation_threshold=30):
        self.rare_trim_threshold  = rare_trim_threshold
        self.rare_model_threshold = rare_model_threshold
        self.rare_generation_threshold = rare_generation_threshold
        self.rare_model_generation_threshold = rare_model_generation_threshold

    @staticmethod
    def _clean_trim(series):
        out = series.fillna('Unspecified').astype(str).str.strip()
        out = out.replace({'': 'Unspecified', 'nan': 'Unspecified', 'None': 'Unspecified'})
        out = out.replace({'Base Grade': 'Unspecified'})
        return out

    @staticmethod
    def _clean_generation(series):
        out = series.fillna('Unspecified').astype(str).str.strip()
        out = out.replace({'': 'Unspecified', 'nan': 'Unspecified', 'None': 'Unspecified'})
        return out

    def fit(self, X, y=None):
        X = X.copy()
        non_ev = X['is_electric'] != 1
        bm = X.loc[non_ev].groupby(['brand', 'model'])['Engine_CC_Clean'].median()
        b  = X.loc[non_ev].groupby('brand')['Engine_CC_Clean'].median()
        self.engine_bm_median_     = bm.to_dict()
        self.engine_b_median_      = b.to_dict()
        self.engine_global_median_ = float(X.loc[non_ev, 'Engine_CC_Clean'].median())

        trim        = self._clean_trim(X['trim_grade_s4'])
        trim_counts = trim.value_counts()
        self.common_trims_ = set(trim_counts[trim_counts >= self.rare_trim_threshold].index)

        generation = self._clean_generation(X['generation'])
        gen_counts = generation.value_counts()
        self.common_generations_ = set(gen_counts[gen_counts >= self.rare_generation_threshold].index)

        pair_counts        = X.groupby(['brand', 'model']).size()
        self.common_pairs_ = set(pair_counts[pair_counts >= self.rare_model_threshold].index)

        bmg = X['brand'].astype(str) + ' | ' + X['model'].astype(str) + ' | ' + generation
        bmg_counts = bmg.value_counts()
        self.common_brand_model_generations_ = set(
            bmg_counts[bmg_counts >= self.rare_model_generation_threshold].index
        )
        return self

    def transform(self, X):
        X = X.copy()
        def fill_engine(row):
            val = row['Engine_CC_Clean']
            if pd.notna(val) or row['is_electric'] == 1:
                return val
            key = (row['brand'], row['model'])
            if key in self.engine_bm_median_ and pd.notna(self.engine_bm_median_[key]):
                return self.engine_bm_median_[key]
            if row['brand'] in self.engine_b_median_ and pd.notna(self.engine_b_median_[row['brand']]):
                return self.engine_b_median_[row['brand']]
            return self.engine_global_median_

        X['Engine_CC_Clean'] = X.apply(fill_engine, axis=1)

        X['trim_grade_s4']   = self._clean_trim(X['trim_grade_s4'])
        X.loc[~X['trim_grade_s4'].isin(self.common_trims_), 'trim_grade_s4'] = 'Other_Trim'
        X['trim_tier_s4']    = X['trim_tier_s4'].fillna('Unspecified').astype(str).str.strip()
        X['trim_tier_s4']    = X['trim_tier_s4'].replace({'': 'Unspecified', 'nan': 'Unspecified', 'None': 'Unspecified'})

        X['generation'] = self._clean_generation(X['generation'])
        X.loc[~X['generation'].isin(self.common_generations_), 'generation'] = 'Other_Generation'

        X['model_s4']        = X['model']
        pair_index  = pd.MultiIndex.from_frame(X[['brand', 'model']])
        common_mask = pair_index.isin(self.common_pairs_)
        X.loc[~common_mask, 'model_s4'] = 'Other'

        X['brand_model_generation'] = X['brand'].astype(str) + ' | ' + X['model'].astype(str) + ' | ' + X['generation']
        X.loc[~X['brand_model_generation'].isin(self.common_brand_model_generations_), 'brand_model_generation'] = 'Other_Brand_Model_Generation'

        return X[NUMERIC_COLS + TARGET_ENCODE_COLS + OHE_COLS + BINARY_COLS]

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols      = cols
        self.smoothing = smoothing

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index, name='target')
        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        for col in self.cols:
            stats  = y.groupby(X[col]).agg(['mean', 'count'])
            smooth = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_) / (stats['count'] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps_[col]).fillna(self.global_mean_).astype(float)
        return X


# ══════════════════════════════════════════════════════════════
#  DATA / MODEL LOADING
# ══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "lgbm.pkl"))

@st.cache_data
def load_dropdown_data():
    with open(os.path.join(BASE_DIR, "dropdown_data_with_generations.json"), "r") as f:
        return json.load(f)

model    = load_model()
dd       = load_dropdown_data()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

JAPANESE  = {'Toyota','Honda','Suzuki','Daihatsu','Nissan','Mitsubishi','Mazda','Subaru','Isuzu','Lexus'}
KOREAN    = {'Hyundai','Kia','SsangYong','Daewoo'}
CHINESE   = {'Changan','Haval','MG','FAW','BAIC','Chery','DFSK','Proton','Jetour','BYD','Deepal','Seres',
             'ORA','Jaecoo','Dongfeng','Forthing','GAC','JMEV','ZOTYE','Daehan','Rinco','Honri'}
EUROPEAN  = {'Mercedes Benz','BMW','Audi','Porsche','Volkswagen','Peugeot','Skoda','Volvo','Bentley',
             'Bugatti','Fiat','Chrysler','Jaguar','Land Rover','Range Rover','MINI'}
AMERICAN  = {'Ford','Chevrolet','Jeep','Dodge','Tesla','GMC','Hummer','Cadillac','Buick'}
PAKISTANI = {'Prince','United','Sogo','Adam','Power','Master','GUGO','Inverex'}
MALAYSIAN = {'Perodua','Proton'}

def get_brand_origin(brand):
    if brand in JAPANESE:  return 'Japanese'
    if brand in KOREAN:    return 'Korean'
    if brand in CHINESE:   return 'Chinese'
    if brand in EUROPEAN:  return 'European'
    if brand in AMERICAN:  return 'American'
    if brand in PAKISTANI: return 'Pakistani'
    if brand in MALAYSIAN: return 'Malaysian'
    return 'Other'

def get_city_tier(city):
    if city in ('Karachi','Lahore','Islamabad'):
        return 'Tier_1'
    if city in ('Rawalpindi','Faisalabad','Peshawar','Multan','Gujranwala','Sialkot','Hyderabad','Quetta'):
        return 'Tier_2'
    return 'Tier_3'

PERFORMANCE_TERMS = ['RS Turbo','RS','AMG','C63','G63','C63 AMG','E63 AMG','SVR','Evo','Evolution','GR Sport','M Series']
PREMIUM_TERMS     = ['Grande','Altis Grande','Altis X','Altis','Oriel','Aspire','VXL','GLS','Prosmatec',
                     'Limited','Signature','FutureSense','High Grade']
MID_TERMS         = ['VXR','GLi','VTi','EXi','DLX','AWD','FWD','HEV','Hybrid']
BASE_TERMS        = ['VX','XLi','XE','Standard','Base','GL','G','X','F','S','L']

def get_trim_tier(trim_grade):
    if not trim_grade or trim_grade in ('Unspecified', ''):
        return 'Unspecified'
    for t in PERFORMANCE_TERMS:
        if t.lower() in trim_grade.lower():
            return 'performance'
    for t in PREMIUM_TERMS:
        if t.lower() in trim_grade.lower():
            return 'premium'
    for t in MID_TERMS:
        if t.lower() in trim_grade.lower():
            return 'mid'
    for t in BASE_TERMS:
        if t.lower() in trim_grade.lower():
            return 'base'
    return 'Unspecified'

def get_price_margin(price_lacs):
    if price_lacs <= 20:   return 0.102
    if price_lacs <= 50:   return 0.055
    if price_lacs <= 100:  return 0.037
    return 0.0475

def get_confidence(brand, price_lacs):
    high_conf_brands = {'Toyota','Honda','Suzuki','Kia','Hyundai'}
    if brand in high_conf_brands and price_lacs <= 100:
        return "high", "High Confidence", "conf-high"
    mainstream = JAPANESE | KOREAN
    if brand in mainstream and price_lacs <= 200:
        return "good", "Good Estimate", "conf-good"
    return "mod", "Moderate Estimate", "conf-mod"

def format_urdu_price(price_lacs):
    return f"{price_lacs:.1f} لاکھ روپے"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def get_scoped_model_options(brand, model_name, generation):
    if not brand or not model_name:
        return {}
    model_options = dd.get(brand, {}).get(model_name, {})
    if generation and generation != "Unspecified":
        return model_options.get("by_generation", {}).get(generation, model_options)
    return model_options

def predict_price(brand, model_name, generation, trim, engine_cc, fuel_type, transmission, year, mileage, city):
    car_age      = 2026 - year
    city_tier    = get_city_tier(city)
    brand_origin = get_brand_origin(brand)
    is_electric  = 1 if fuel_type in ('Electric', 'REEV') else 0
    trim_tier    = get_trim_tier(trim)
    trim_grade   = trim if trim and trim != 'Unspecified' else 'Unspecified'

    row = pd.DataFrame([{
        'car_age':        car_age,
        'Mileage':        mileage,
        'Engine_CC_Clean': float(engine_cc) if engine_cc else np.nan,
        'brand':          brand,
        'model':          model_name,
        'generation':     generation,
        'Fuel_Type':      fuel_type,
        'Transmission':   transmission,
        'brand_origin':   brand_origin,
        'city_tier':      city_tier,
        'trim_tier_s4':   trim_tier,
        'trim_grade_s4':  trim_grade,
        'is_electric':    is_electric,
    }])

    log_pred  = model.predict(row)[0]
    price_lac = np.expm1(log_pred)
    return price_lac

def parse_listing_card(car, brand, model_name):
    listing_url = ""
    cover_url = ""

    script = car.find("script", type="application/ld+json")
    if script and script.string:
        try:
            ld = json.loads(script.string)
            listing_url = ld.get("offers", {}).get("url", "") or ""
            cover_url = ld.get("image", "") or ""
        except Exception:
            pass

    if not listing_url:
        a_tag = car.find("a", href=True)
        if a_tag:
            href = a_tag["href"]
            listing_url = href if href.startswith("http") else f"https://www.pakwheels.com{href}"

    if not cover_url:
        img = car.find("img")
        if img:
            cover_url = img.get("data-original") or img.get("data-src") or img.get("src") or ""

    h3 = car.find("h3")
    title = h3.get_text(" ", strip=True) if h3 else f"{brand} {model_name}"

    price_div = car.find("div", class_=lambda x: x and "price-details" in x)
    price = price_div.get_text(" ", strip=True) if price_div else ""

    city = ""
    city_ul = car.find("ul", class_="search-vehicle-info")
    if city_ul:
        city_li = city_ul.find("li")
        if city_li:
            city = city_li.get_text(" ", strip=True)

    year_val = ""
    specs_ul = car.find("ul", class_="search-vehicle-info-2")
    if specs_ul:
        items = [li.get_text(" ", strip=True) for li in specs_ul.find_all("li")]
        if items:
            year_val = items[0]

    return {
        "brand": brand,
        "model": model_name,
        "Title": title,
        "City": city,
        "Year": year_val,
        "Price": price,
        "Listing_URL": listing_url,
        "Cover_URL": cover_url,
    }

def parse_price_lacs(price_text):
    text = str(price_text).lower().replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    value = float(match.group(1))
    if "crore" in text:
        return value * 100
    if "lac" in text or "lakh" in text:
        return value
    return value

def generation_matches_title(generation, title):
    if not generation or generation == "Unspecified":
        return False
    primary = str(generation).split("/")[0].strip().lower()
    return bool(primary and primary in str(title).lower())

@st.cache_data(ttl=900, show_spinner=False)
def fetch_similar_listing_page(brand, model_name, trim, page_no):
    query_bits = [brand, model_name]
    if trim and trim != "Unspecified":
        query_bits.append(trim)
    query = quote_plus(" ".join(query_bits))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    url = f"https://www.pakwheels.com/used-cars/search/-/?q={query}&page={page_no}"
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.find_all("li", class_=lambda x: x and "classified-listing" in x)
    rows = []
    for card in cards:
        row = parse_listing_card(card, brand, model_name)
        title_lower = row["Title"].lower()
        if brand.lower() in title_lower and model_name.split()[0].lower() in title_lower:
            rows.append(row)
    return pd.DataFrame(rows)

def rank_similar_listings(listings_df, generation, city, user_year, predicted_price, low_price, high_price):
    if listings_df.empty:
        return pd.DataFrame()

    df = listings_df.drop_duplicates(subset=["Listing_URL"]).copy()
    df["Price_Lacs"] = df["Price"].apply(parse_price_lacs)
    df["Year_Num"] = pd.to_numeric(df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    df["City_Match"] = df["City"].astype(str).str.lower().eq(city.lower())
    df["Generation_Match"] = df["Title"].apply(lambda title: generation_matches_title(generation, title))
    df["Price_Match"] = df["Price_Lacs"].between(float(low_price), float(high_price), inclusive="both")
    df["Year_Match"] = df["Year_Num"].eq(int(user_year))
    df["Hybrid_Match"] = df["Price_Match"] & df["Year_Match"]
    df["Price_Diff"] = (df["Price_Lacs"] - float(predicted_price)).abs()
    df["Year_Diff"] = (df["Year_Num"] - int(user_year)).abs()
    df = df.sort_values(
        ["Hybrid_Match", "Price_Diff", "Year_Diff", "City_Match", "Generation_Match"],
        ascending=[False, True, True, False, False],
        na_position="last",
    )
    return df.drop(
        columns=[
            "Price_Lacs", "Year_Num", "City_Match", "Generation_Match",
            "Price_Match", "Year_Match", "Hybrid_Match", "Price_Diff", "Year_Diff"
        ],
        errors="ignore",
    ).reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def scrape_search_listings(brand, model_name, generation, trim, city, user_year, max_pages=5):
    query_bits = [brand, model_name]
    if trim and trim != "Unspecified":
        query_bits.append(trim)
    query_bits.append(str(user_year))
    if city and city != "Other":
        query_bits.append(city)
    query = quote_plus(" ".join(query_bits))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    rows = []
    for page_no in range(1, max_pages + 1):
        url = f"https://www.pakwheels.com/used-cars/search/-/?q={query}&page={page_no}"
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.find_all("li", class_=lambda x: x and "classified-listing" in x)
        for card in cards:
            row = parse_listing_card(card, brand, model_name)
            title_lower = row["Title"].lower()
            if brand.lower() in title_lower and model_name.split()[0].lower() in title_lower:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["Listing_URL"])
    df["Year_Num"] = pd.to_numeric(df["Year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    df["City_Match"] = df["City"].astype(str).str.lower().eq(city.lower())
    df["Generation_Match"] = df["Title"].apply(lambda title: generation_matches_title(generation, title))
    df = df[df["Year_Num"].eq(int(user_year))]
    if city and city != "Other":
        df = df[df["City_Match"]]
    if df.empty:
        return pd.DataFrame()
    df["Year_Diff"] = (df["Year_Num"] - int(user_year)).abs()
    df = df.sort_values(["Generation_Match", "Year_Diff", "City_Match"], ascending=[False, True, False], na_position="last")
    return df.drop(columns=["Year_Num", "City_Match", "Generation_Match", "Year_Diff"], errors="ignore").reset_index(drop=True)

def render_car_card(row):
    cover  = escape(str(row.get('Cover_URL', '')).strip(), quote=True)
    url    = escape(str(row.get('Listing_URL', '#')).strip() or "#", quote=True)
    year   = escape(str(row.get('Year', '')).strip())
    city   = escape(str(row.get('City', '')).strip())
    brand  = str(row.get('brand', '')).strip()
    model_ = str(row.get('model', '')).strip()
    title  = escape(str(row.get('Title', '')).strip() or f"{brand} {model_}")
    price  = escape(str(row.get('Price', '')).strip())

    if cover and cover not in ('nan', '', 'None'):
        img_html = f'<div class="car-card-img-wrap"><img class="car-card-img" src="{cover}" onerror="this.parentElement.innerHTML=\'<div class=car-card-placeholder>🚗</div>\'"></div>'
    else:
        img_html = '<div class="car-card-placeholder">🚗</div>'

    st.markdown(f"""
    <a href="{url}" target="_blank" class="car-card-wrap">
        <div class="car-card">
            {img_html}
            <div class="car-card-body">
                <div class="car-card-title">{title}</div>
                <div style="font-family:Rajdhani,sans-serif; color:#FF4B4B; font-weight:700; font-size:1rem; margin-bottom:5px;">{price}</div>
                <div class="car-card-meta">📅 <span>{year}</span> &nbsp;📍 <span>{city}</span></div>
                <div class="car-card-link">View on PakWheels →</div>
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

def render_listing_pages(listings_df, page_key, button_prefix):
    page_size = 5
    total = len(listings_df)
    total_pages = max(1, int(np.ceil(total / page_size)))
    current_page = min(st.session_state.get(page_key, 0), total_pages - 1)
    start = current_page * page_size
    page_rows = listings_df.iloc[start:start + page_size]

    anim_name = f"cardSlideIn_{page_key}_{current_page}"
    st.markdown(f"""
    <style>
    @keyframes {anim_name} {{
        0%   {{ opacity: 0; transform: translateY(18px) scale(0.97); }}
        60%  {{ opacity: 1; transform: translateY(-3px) scale(1.005); }}
        100% {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    #cards-page-{page_key}-{current_page} .car-card {{
        animation: {anim_name} 0.38s cubic-bezier(0.22, 0.61, 0.36, 1) both;
    }}
    #cards-page-{page_key}-{current_page} .car-card:nth-child(1) {{ animation-delay: 0.00s; }}
    #cards-page-{page_key}-{current_page} .car-card:nth-child(2) {{ animation-delay: 0.06s; }}
    #cards-page-{page_key}-{current_page} .car-card:nth-child(3) {{ animation-delay: 0.12s; }}
    #cards-page-{page_key}-{current_page} .car-card:nth-child(4) {{ animation-delay: 0.18s; }}
    #cards-page-{page_key}-{current_page} .car-card:nth-child(5) {{ animation-delay: 0.24s; }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<p style="color:#666; font-size:0.78rem; margin-bottom:1rem;">Showing {start + 1}-{start + len(page_rows)} of {total} relevant listings</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="cards-animated" id="cards-page-{page_key}-{current_page}">', unsafe_allow_html=True)
    card_cols = st.columns(len(page_rows))
    for i, (_, row) in enumerate(page_rows.iterrows()):
        with card_cols[i]:
            render_car_card(row)
    st.markdown('</div>', unsafe_allow_html=True)

    prev_col, mid_col, next_col = st.columns([1, 3, 1])
    with prev_col:
        if current_page > 0 and st.button("← Previous", key=f"{button_prefix}_prev", use_container_width=True):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with mid_col:
        st.markdown(f'<div style="text-align:center; color:#666; font-size:0.78rem; padding-top:0.65rem;">Page {current_page + 1} of {total_pages}</div>', unsafe_allow_html=True)
    with next_col:
        if current_page < total_pages - 1 and st.button("Next →", key=f"{button_prefix}_next", use_container_width=True):
            st.session_state[page_key] = current_page + 1
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="pw-header">
    <div>
        <div class="pw-logo">pakistani Cars <span>Price Predictor</span></div>
        <div class="pw-tagline">2026 Research Study · 58,750 Listings · LightGBM</div>
    </div>
    <div class="pw-badge">R² 0.9676</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

PAGES = ["🔮 Predict Price", "🔎 Search Cars", "📊 Market Insights", "ℹ️ About This Study", "👤 About the Author"]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 24px 0 20px;'>
        <div style='font-family: Rajdhani, sans-serif; font-size:1.1rem; color:#FF4B4B; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>Pakistani Cars Price AI</div>
        <div style='font-size:2.2rem; margin-bottom:10px;'>🚗</div>
        <div style='font-family: Rajdhani, sans-serif; font-size:1.2rem; color:#FF4B4B; font-weight:700; letter-spacing:0.08em;'>Navigation</div>
        <div style='width:30px; height:1px; background:#FF4B4B; margin:8px auto; opacity:0.4;'></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("page", PAGES, label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 14px 0; text-align:center;'>
        <div style='font-size:0.68rem; color:#888; letter-spacing:0.16em; text-transform:uppercase; line-height:2.2;'>
            Powered by<br>
            <span style='color:#FF4B4B; font-weight:700;'>LightGBM · scikit-learn</span><br>
            <span style='color:#FF4B4B; font-weight:700;'>Streamlit · PakWheels Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 10px 0; text-align:center;'>
        <div style='font-size:0.65rem; color:#555; line-height:1.8;'>
            Prices are predicted listing prices.<br>
            Not transaction prices.<br>
            Use as reference only.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT PRICE
# ══════════════════════════════════════════════════════════════

if page == "🔮 Predict Price":
    st.markdown('<div class="section-heading">Predict Used Car Price</div><div class="section-divider"></div>', unsafe_allow_html=True)

    brands_list = sorted(dd.keys())

    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-label">Vehicle Identity</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", brands_list)
    with col2:
        models_list = sorted(dd[brand].keys()) if brand else []
        model_name  = st.selectbox("Model", models_list)

    col1b, col2b = st.columns(2)
    with col1b:
        if brand and model_name:
            gens_raw = dd[brand][model_name].get("generations", [])
            extracted_gens = [g['generation'] for g in gens_raw if g and 'generation' in g]
            unique_gens = list(dict.fromkeys(extracted_gens))
            unique_gens = sorted(unique_gens, key=natural_sort_key)
            gens_list = ["Unspecified"] + unique_gens
        else:
            gens_list = ["Unspecified"]
        generation = st.selectbox("Generation", gens_list)

    scoped_options = get_scoped_model_options(brand, model_name, generation)

    with col2b:
        if scoped_options:
            trims_raw = scoped_options.get("trims", [])
            trims_list = ["Unspecified"] + [t for t in trims_raw if t]
        else:
            trims_list = ["Unspecified"]
        trim = st.selectbox("Trim / Variant", trims_list)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-label">Technical Specs</div>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        if scoped_options:
            engines_raw = scoped_options.get("engines", [])
            engines_list = []
            for e in engines_raw:
                if e not in (None, '', 'nan') and str(e).lower() != 'nan':
                    try:
                        engines_list.append(int(float(e)))
                    except ValueError:
                        engines_list.append(e)
            engines_list = sorted(list(set(engines_list)))
        else:
            engines_list = []
            
        if engines_list:
            engine_cc = st.selectbox("Engine CC", engines_list)
        else:
            # Fallback to standard market engine sizes instead of allowing absurd values
            standard_engines = [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000, 2400, 2500, 2700, 2800, 3000, 4000]
            engine_cc = st.selectbox("Engine CC", standard_engines, index=4)

    with col5:
        if scoped_options:
            fuels_list = scoped_options.get("fuels", [])
            fuels_list = [f for f in fuels_list if f not in (None, '', 'nan') and str(f).lower() != 'nan']
        else:
            fuels_list = []
        if not fuels_list:
            fuels_list = ["Petrol", "Hybrid", "Diesel", "Electric", "REEV", "CNG"]
        fuel_type = st.selectbox("Fuel Type", fuels_list)

    with col6:
        if scoped_options:
            trans_list = scoped_options.get("transmissions", [])
            trans_list = [t for t in trans_list if t not in (None, '', 'nan') and str(t).lower() != 'nan']
        else:
            trans_list = []
        if not trans_list:
            trans_list = ["Automatic", "Manual"]
        transmission = st.selectbox("Transmission", trans_list)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-label">Condition & Location</div>', unsafe_allow_html=True)

    col7, col8, col9 = st.columns(3)
    with col7:
        years_list = sorted(scoped_options.get("years", []), reverse=True) if scoped_options else []
        if not years_list:
            years_list = list(range(2026, 1999, -1))
        year = st.selectbox("Model Year", years_list)
    with col8:
        # Dynamic default mileage (approx 12,000 km per year)
        car_age_approx = max(1, 2026 - year)
        default_mileage = int(car_age_approx * 12000)
        
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=default_mileage, step=1000)
    with col9:
        city = st.selectbox("City", [
            "Karachi","Lahore","Islamabad","Rawalpindi","Faisalabad",
            "Peshawar","Multan","Gujranwala","Sialkot","Hyderabad","Quetta","Other"
        ])

    st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5])
    with col_btn2:
        predict_clicked = st.button("⚡ PREDICT PRICE", use_container_width=True, type="primary")

    if predict_clicked:
        if not brand or not model_name:
            st.warning("Please select a brand and model first.")
        else:
            with st.spinner("Running model..."):
                try:
                    price = predict_price(brand, model_name, generation, trim, engine_cc,
                                          fuel_type, transmission, year, mileage, city)
                    margin    = get_price_margin(price)
                    low       = price * (1 - margin)
                    high      = price * (1 + margin)
                    conf_key, conf_label, conf_cls = get_confidence(brand, price)
                    urdu_str  = format_urdu_price(price)
                    st.session_state["last_prediction_inputs"] = {
                        "brand": brand,
                        "model_name": model_name,
                        "generation": generation,
                        "trim": trim,
                        "city": city,
                        "year": year,
                        "predicted_price": price,
                        "low": low,
                        "high": high,
                    }
                    st.session_state["last_prediction_result"] = {
                        "price": price,
                        "low": low,
                        "high": high,
                        "urdu_str": urdu_str,
                        "confidence_label": conf_label,
                        "confidence_cls": conf_cls,
                    }
                    st.session_state["show_live_listings"] = False
                    st.session_state["live_listings_page"] = 0
                    st.session_state["similar_source_page"] = 1
                    st.session_state["similar_source_exhausted"] = False
                    st.session_state["similar_listing_rows"] = []

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Estimated Market Price</div>
                        <div class="result-price">PKR {price:.1f} Lacs</div>
                        <div class="result-range">Range: PKR {low:.1f} — {high:.1f} Lacs</div>
                        <div class="result-urdu">{urdu_str}</div>
                        <span class="confidence-badge {conf_cls}">{conf_label}</span>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    result_inputs = st.session_state.get("last_prediction_inputs")
    if result_inputs:
        result_values = st.session_state.get("last_prediction_result")
        if result_values and not predict_clicked:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">PKR {result_values["price"]:.1f} Lacs</div>
                <div class="result-range">Range: PKR {result_values["low"]:.1f} - {result_values["high"]:.1f} Lacs</div>
                <div class="result-urdu">{result_values["urdu_str"]}</div>
                <span class="confidence-badge {result_values["confidence_cls"]}">{result_values["confidence_label"]}</span>
            </div>
            """, unsafe_allow_html=True)

        col_live1, col_live2, col_live3 = st.columns([1.5, 2, 1.5])
        with col_live2:
            if st.button("SHOW SIMILAR LISTINGS", use_container_width=True):
                st.session_state["show_live_listings"] = True
                st.session_state["live_listings_page"] = 0
                st.session_state["similar_source_page"] = 1
                st.session_state["similar_source_exhausted"] = False
                st.session_state["similar_listing_rows"] = []

        if st.session_state.get("show_live_listings"):
            st.markdown('<div class="section-heading" style="font-size:1.3rem; margin-top:2rem;">Similar Listings on PakWheels</div><div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888; font-size:0.8rem; margin-bottom:1rem;">Searching for closest matching cars. Please wait...</p>', unsafe_allow_html=True)
            with st.spinner("Fetching fresh listings..."):
                try:
                    if not st.session_state.get("similar_listing_rows"):
                        first_page = fetch_similar_listing_page(
                            result_inputs["brand"],
                            result_inputs["model_name"],
                            result_inputs["trim"],
                            st.session_state.get("similar_source_page", 1),
                        )
                        st.session_state["similar_listing_rows"] = first_page.to_dict("records")
                        st.session_state["similar_source_page"] = st.session_state.get("similar_source_page", 1) + 1
                        if first_page.empty:
                            st.session_state["similar_source_exhausted"] = True

                    loaded = pd.DataFrame(st.session_state.get("similar_listing_rows", []))
                    similar = rank_similar_listings(
                        loaded,
                        result_inputs["generation"],
                        result_inputs["city"],
                        result_inputs["year"],
                        result_inputs["predicted_price"],
                        result_inputs["low"],
                        result_inputs["high"],
                    )
                    if not similar.empty:
                        page_size = 5
                        current_page = st.session_state.get("live_listings_page", 0)
                        start = current_page * page_size
                        page_rows = similar.iloc[start:start + page_size]

                        live_anim = f"cardSlideInLive_{current_page}"
                        st.markdown(f"""
                        <style>
                        @keyframes {live_anim} {{
                            0%   {{ opacity: 0; transform: translateY(18px) scale(0.97); }}
                            60%  {{ opacity: 1; transform: translateY(-3px) scale(1.005); }}
                            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
                        }}
                        #cards-live-{current_page} .car-card {{
                            animation: {live_anim} 0.38s cubic-bezier(0.22, 0.61, 0.36, 1) both;
                        }}
                        #cards-live-{current_page} .car-card:nth-child(1) {{ animation-delay: 0.00s; }}
                        #cards-live-{current_page} .car-card:nth-child(2) {{ animation-delay: 0.06s; }}
                        #cards-live-{current_page} .car-card:nth-child(3) {{ animation-delay: 0.12s; }}
                        #cards-live-{current_page} .car-card:nth-child(4) {{ animation-delay: 0.18s; }}
                        #cards-live-{current_page} .car-card:nth-child(5) {{ animation-delay: 0.24s; }}
                        </style>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<p style="color:#666; font-size:0.78rem; margin-bottom:1rem;">Showing {start + 1}-{start + len(page_rows)} of {len(similar)} loaded listings</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="cards-animated" id="cards-live-{current_page}">', unsafe_allow_html=True)
                        card_cols = st.columns(len(page_rows))
                        for i, (_, row) in enumerate(page_rows.iterrows()):
                            with card_cols[i]:
                                render_car_card(row)
                        st.markdown('</div>', unsafe_allow_html=True)

                        prev_col, mid_col, next_col = st.columns([1, 3, 1])
                        with prev_col:
                            if current_page > 0 and st.button("← Previous", key="similar_listings_prev", use_container_width=True):
                                st.session_state["live_listings_page"] = current_page - 1
                                st.rerun()
                        with mid_col:
                            st.markdown(f'<div style="text-align:center; color:#666; font-size:0.78rem; padding-top:0.65rem;">Page {current_page + 1}</div>', unsafe_allow_html=True)
                        with next_col:
                            has_loaded_next = len(similar) > (current_page + 1) * page_size
                            can_fetch_more = (
                                not st.session_state.get("similar_source_exhausted", False)
                                and st.session_state.get("similar_source_page", 1) <= 5
                            )
                            if (has_loaded_next or can_fetch_more) and st.button("Next →", key="similar_listings_next", use_container_width=True):
                                should_advance = has_loaded_next
                                if not has_loaded_next:
                                    next_page = fetch_similar_listing_page(
                                        result_inputs["brand"],
                                        result_inputs["model_name"],
                                        result_inputs["trim"],
                                        st.session_state.get("similar_source_page", 1),
                                    )
                                    if next_page.empty:
                                        st.session_state["similar_source_exhausted"] = True
                                    else:
                                        combined = pd.concat([loaded, next_page], ignore_index=True)
                                        st.session_state["similar_listing_rows"] = combined.drop_duplicates(subset=["Listing_URL"]).to_dict("records")
                                        st.session_state["similar_source_page"] = st.session_state.get("similar_source_page", 1) + 1
                                        should_advance = True
                                if st.session_state.get("similar_source_page", 1) > 5:
                                    st.session_state["similar_source_exhausted"] = True
                                if should_advance:
                                    st.session_state["live_listings_page"] = current_page + 1
                                st.rerun()
                    else:
                        st.markdown('<p style="color:#555; font-size:0.82rem; margin-top:1rem;">No matching live listings found right now.</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not fetch live PakWheels listings right now: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — SEARCH CARS
# ══════════════════════════════════════════════════════════════

elif page == "🔎 Search Cars":
    st.markdown('<div class="section-heading">Search Cars</div><div class="section-divider"></div>', unsafe_allow_html=True)
    brands_list = sorted(dd.keys())

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        search_brand = st.selectbox("Search Brand", brands_list, key="search_brand")
    with search_col2:
        search_models = sorted(dd[search_brand].keys()) if search_brand else []
        search_model = st.selectbox("Search Model", search_models, key="search_model")

    search_col3, search_col4, search_col5 = st.columns(3)
    with search_col3:
        if search_brand and search_model:
            search_gens_raw = dd[search_brand][search_model].get("generations", [])
            search_gens = [g["generation"] for g in search_gens_raw if g and "generation" in g]
            search_gens = sorted(list(dict.fromkeys(search_gens)), key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])
            search_gens = ["Unspecified"] + search_gens
        else:
            search_gens = ["Unspecified"]
        search_generation = st.selectbox("Search Generation", search_gens, key="search_generation")

    # Scope all dependent fields to the selected generation
    search_scoped = get_scoped_model_options(search_brand, search_model, search_generation)

    with search_col4:
        if search_scoped:
            search_trims_raw = search_scoped.get("trims", [])
            search_trims = ["Unspecified"] + [t for t in search_trims_raw if t]
        else:
            search_trims = ["Unspecified"]
        search_trim = st.selectbox("Search Variant", search_trims, key="search_trim")
    with search_col5:
        search_years_list = sorted(search_scoped.get("years", []), reverse=True) if search_scoped else []
        if not search_years_list:
            search_years_list = list(range(2026, 1999, -1))
        search_year = st.selectbox("Search Year", search_years_list, key="search_year")

    search_col6, search_col7, search_col8 = st.columns([1, 1, 1])
    with search_col6:
        search_city = st.selectbox("Search City", [
            "Karachi","Lahore","Islamabad","Rawalpindi","Faisalabad",
            "Peshawar","Multan","Gujranwala","Sialkot","Hyderabad","Quetta","Other"
        ], key="search_city")
    with search_col7:
        st.write("")
    with search_col8:
        st.write("")
        if st.button("SEARCH CARS", key="search_cars_btn", use_container_width=True):
            st.session_state["search_cars_inputs"] = {
                "brand": search_brand,
                "model_name": search_model,
                "generation": search_generation,
                "trim": search_trim,
                "city": search_city,
                "year": search_year,
            }
            st.session_state["search_cars_page"] = 0

    search_inputs = st.session_state.get("search_cars_inputs")
    if search_inputs:
        st.markdown('<p style="color:#888; font-size:0.8rem; margin:1.2rem 0 1rem;">Live PakWheels search matched by brand, model, generation, variant, year, and city.</p>', unsafe_allow_html=True)
        with st.spinner("Searching live PakWheels listings..."):
            try:
                search_results = scrape_search_listings(
                    search_inputs["brand"],
                    search_inputs["model_name"],
                    search_inputs["generation"],
                    search_inputs["trim"],
                    search_inputs["city"],
                    search_inputs["year"],
                )
                if not search_results.empty:
                    render_listing_pages(search_results, "search_cars_page", "search_cars")
                else:
                    st.markdown('<p style="color:#555; font-size:0.82rem; margin-top:1rem;">No live listings matched this exact search right now.</p>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not search PakWheels listings right now: {e}")


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════

elif page == "📊 Market Insights":
    st.markdown('<div class="section-heading">Market Insights</div><div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888; font-size:0.88rem; margin-bottom:2rem; max-width:720px;">Key findings from the 2026 research study on Pakistani used car pricing. Based on 58,750 PakWheels listings from the final modelling dataset.</p>', unsafe_allow_html=True)

    charts = [
        {
            "path": "eda/19_market_shift_comparison.png",
            "title": "2020 vs 2026 Price Shift",
            "caption": "Prices of identical models are 65–197% higher in 2026 vs 2020, driven by PKR depreciation and import duty restructuring. ML studies trained on the 2020 Kaggle extract are structurally outdated for today's market."
        },
        {
            "path": "eda/02_median_price_by_year.png",
            "title": "Median Price by Model Year",
            "caption": "Car prices escalated sharply after 2021 due to import disruptions, PKR depreciation against the USD, and successive duty restructuring cycles."
        },
        {
            "path": "eda/09_brand_origin_price_boxplot.png",
            "title": "Price Distribution by Brand Origin",
            "caption": "European and American brands command the highest median prices. Chinese brands show lower resale retention at this stage of Pakistani market penetration — reflecting uncertainty about long-term parts availability."
        },
    ]

    for chart in charts:
        st.markdown(f'<div class="insight-card">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:Rajdhani,sans-serif; font-size:1rem; font-weight:700; color:#FF4B4B; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px;">{chart["title"]}</div>', unsafe_allow_html=True)
        abs_path = os.path.join(BASE_DIR, chart["path"])
        if os.path.exists(abs_path):
            st.image(abs_path, use_container_width=True)
        else:
            st.markdown(f'<div style="color:#555; font-size:0.82rem; padding:20px;">Chart not found: {chart["path"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-caption">{chart["caption"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════

elif page == "ℹ️ About This Study":
    st.markdown('<div class="section-heading">About This Study</div><div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">📄 The Research</div>
            <p style='color:#F0F0F0; font-size:0.95rem; line-height:1.9; margin-bottom:14px;'>
                <strong style='color:#FF4B4B;'>Pakistani Used Car Price Prediction — 2026</strong>
            </p>
            <p style='color:#888; font-size:0.86rem; line-height:1.85; margin-bottom:12px;'>
                This is the first Pakistani used car pricing study built on <strong style='color:#BBBBBB;'>2025–2026 PakWheels data</strong>.
                Prior studies (Asghar et al. 2021, Ahtesham & Zulfiqar 2022) relied on a 2020 Kaggle extract
                where prices are <strong style='color:#FF4B4B;'>65–197% below</strong> current market values —
                making those models structurally obsolete for today's buyers and sellers.
            </p>
            <p style='color:#888; font-size:0.86rem; line-height:1.85; margin:0;'>
                This study introduces <strong style='color:#BBBBBB;'>Generation & trim grade hierarchy</strong> as features for the first time
                in the Pakistani context — capturing the significant price variance between e.g. a Corolla XLi and a Corolla Altis Grande , Honda Civic Reborn and Honda Civic 11 Generation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">👥 Author</div>
            <div class="about-stat">
                <div class="about-stat-icon">🎓</div>
                <div>
                    <div class="about-stat-label">Author</div>
                    <div class="about-stat-value">Muhammad Haris Afridi</div>
                </div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">🏛️</div>
                <div>
                    <div class="about-stat-label">Institution</div>
                    <div class="about-stat-value">Dept. of ICS/IT, UAP Peshawar</div>
                </div>
            </div>
            <div class="about-stat" style="border-bottom:none;">
                <div class="about-stat-icon">📰</div>
                <div>
                    <div class="about-stat-label">Paper</div>
                    <div class="about-stat-value">ESA — link pending</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns([2, 3])
    with col3:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">⚙️ Model Performance</div>
            <div class="about-stat">
                <div class="about-stat-icon">📊</div>
                <div><div class="about-stat-label">Best Model</div><div class="about-stat-value">LightGBM</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">🎯</div>
                <div><div class="about-stat-label">Test R²</div><div class="about-stat-value">0.9676</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">📉</div>
                <div><div class="about-stat-label">RMSE</div><div class="about-stat-value">5.84 Lacs PKR</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">📐</div>
                <div><div class="about-stat-label">MAE</div><div class="about-stat-value">2.87 Lacs PKR</div></div>
            </div>
            <div class="about-stat" style="border-bottom:none;">
                <div class="about-stat-icon">🔁</div>
                <div><div class="about-stat-label">5-Fold CV R²</div><div class="about-stat-value">0.9666 ± 0.0012</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">🗂️ Dataset & Methodology</div>
            <div class="about-stat">
                <div class="about-stat-icon">📦</div>
                <div><div class="about-stat-label">Source</div><div class="about-stat-value">PakWheels.com — 2025–2026 scrape</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">🔢</div>
                <div><div class="about-stat-label">Listings</div><div class="about-stat-value">58,750 modelling records</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">🏷️</div>
                <div><div class="about-stat-label">Brands Covered</div><div class="about-stat-value">76+ brands, 600+ models</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">🔬</div>
                <div><div class="about-stat-label">Key Novelty</div><div class="about-stat-value">Generation and Trim grade hierarchy as a feature</div></div>
            </div>
            <div class="about-stat">
                <div class="about-stat-icon">📅</div>
                <div><div class="about-stat-label">Prior Studies Gap</div><div class="about-stat-value">2020 data — 65–197% price gap vs 2026</div></div>
            </div>
            <div class="about-stat" style="border-bottom:none;">
                <div class="about-stat-icon">🧪</div>
                <div><div class="about-stat-label">Validation</div><div class="about-stat-value">5-fold stratified cross-validation</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,75,75,0.04); border:1px solid rgba(255,75,75,0.12); border-radius:6px;
                padding:16px 22px; margin-top:1rem; font-size:0.8rem; color:#666; line-height:1.8; text-align:center;">
        ⚠️ Prices shown are predicted <em>listing</em> prices based on PakWheels data.
        Not transaction prices. Actual sale prices may differ.
        Use as a reference point only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — ABOUT THE AUTHOR
# ══════════════════════════════════════════════════════════════

elif page == "👤 About the Author":
    st.markdown('<div class="section-heading">About the Author</div><div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Hero card with avatar + bio
    st.markdown("""
    <div class="author-hero">
        <img class="author-avatar"
             src="https://github.com/harisyar-ai.png"
             alt="Muhammad Haris Afridi"
             onerror="this.src='https://via.placeholder.com/100/1a1a1a/FF4B4B?text=MH'">
        <div class="author-hero-info">
            <div class="author-name">Muhammad Haris Afridi</div>
            <div class="author-role">AI Engineer &amp; Full-Stack Developer</div>
            <div class="author-bio">
                I am a self-taught AI Engineer and Full-Stack Developer from Peshawar, Pakistan.
                I love turning raw ideas into real-world tools — whether it's a machine learning model,
                a government web platform, or an intelligent recommender system.
                Currently pursuing a BS in Artificial Intelligence at Agriculture University of Peshawar
                and serving as a Research Assistant at the Digital Image Processing (DIP) Lab,
                Islamia College Peshawar.
            </div>
            <div class="author-badges">
                <span class="author-badge">🧠 AI Engineer</span>
                <span class="author-badge">💻 Full-Stack Dev</span>
                <span class="author-badge">🎓 BSc AI Student</span>
                <span class="author-badge">🔬 DIP Lab RA</span>
                <span class="author-badge">📚 Avid Reader</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Social links
    st.markdown("""
    <div class="author-socials">
        <a href="https://harisyar-ai.github.io/harisyar-ai/" target="_blank" class="social-pill portfolio">
            <i class="fas fa-globe"></i> Portfolio
        </a>
        <a href="https://github.com/harisyar-ai" target="_blank" class="social-pill github">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a href="https://www.linkedin.com/in/harisyar-ai/" target="_blank" class="social-pill linkedin">
            <i class="fab fa-linkedin-in"></i> LinkedIn
        </a>
        <a href="mailto:mharisyar.ai@gmail.com" class="social-pill email">
            <i class="fas fa-envelope"></i> mharisyar.ai@gmail.com
        </a>
        <a href="https://wa.me/923339342567?text=Hi%20Haris!%20I%20came%20across%20your%20car%20price%20predictor%20and%20would%20love%20to%20connect."
           target="_blank" class="social-pill whatsapp">
            <i class="fab fa-whatsapp"></i> WhatsApp
        </a>
    </div>
    """, unsafe_allow_html=True)

    # ── Info grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="author-info-card">
            <div class="author-info-card-title">🎓 Education &amp; Role</div>
            <div class="author-info-row">
                <div class="author-info-icon">🏛️</div>
                <div>
                    <div class="author-info-label">University</div>
                    <div class="author-info-value">Agriculture University of Peshawar</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">📖</div>
                <div>
                    <div class="author-info-label">Degree</div>
                    <div class="author-info-value">BS Artificial Intelligence (2023–2027)</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">🔬</div>
                <div>
                    <div class="author-info-label">Research Role</div>
                    <div class="author-info-value">Research Assistant, DIP Lab — Islamia College Peshawar</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">📍</div>
                <div>
                    <div class="author-info-label">Location</div>
                    <div class="author-info-value">Peshawar, Khyber Pakhtunkhwa, Pakistan</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="author-info-card">
            <div class="author-info-card-title">🚀 This Project</div>
            <div class="author-info-row">
                <div class="author-info-icon">📊</div>
                <div>
                    <div class="author-info-label">Research Type</div>
                    <div class="author-info-value">Used Car Price Prediction — Pakistan 2026</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">🤖</div>
                <div>
                    <div class="author-info-label">Model Used</div>
                    <div class="author-info-value">LightGBM — R² 0.9676</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">📦</div>
                <div>
                    <div class="author-info-label">Dataset</div>
                    <div class="author-info-value">58,750 PakWheels listings (2025–2026 scrape)</div>
                </div>
            </div>
            <div class="author-info-row">
                <div class="author-info-icon">📰</div>
                <div>
                    <div class="author-info-label">Publication</div>
                    <div class="author-info-value">ESA Submission — link pending</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.6rem; background:rgba(255,75,75,0.04); border:1px solid rgba(255,75,75,0.1);
                border-radius:8px; padding:18px 24px; font-size:0.83rem; color:#888; line-height:1.85;">
        🌐 Want to see more of my work? Visit my portfolio at
        <a href="https://harisyar-ai.github.io/harisyar-ai/" target="_blank"
           style="color:var(--red2); font-weight:600; text-decoration:none; letter-spacing:0.04em;">
           harisyar-ai.github.io/harisyar-ai
        </a>
        — featuring all my projects, skills, certifications, and contact details.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="pw-footer">
    <div class="pw-footer-logo">Pakistani Cars Price AI</div>
    <div class="pw-footer-sub">2026 Research Study · Institute of Computer Sciences and Information Technology, The University of Agriculture, Peshawar, Pakistan</div>
    <div class="pw-footer-copy">
        © 2026 Muhammad Haris Afridi &nbsp;·&nbsp;
        Trained on 58,750 PakWheels listings &nbsp;·&nbsp;
        LightGBM · scikit-learn · Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
