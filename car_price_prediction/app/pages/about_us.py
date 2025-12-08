import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="About Us", page_icon="👤", layout="wide")

# Paths
ASSETS = Path(__file__).parents[1] / "assets"
PROFILE_PATH = ASSETS / "profile.png"  # place the processed image here

# -------------------- CSS + Particles + HTML --------------------
particles_and_style = f"""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap">
<style>
:root {{
    --bg: #0e1117;
    --card: rgba(255,255,255,0.05);
    --accent: #ff4b4b;
    --accent-2: #ff7b00;
    font-family: 'Inter', sans-serif;
}}
html, body {{
    background: var(--bg) !important;
}}

/* Gradient header */
.header {{
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    padding: 28px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 30px rgba(255,75,75,0.08);
}}

/* Glass card */
.card {{
    background: {{"rgba(255,255,255,0.04)"}};
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(6px);
    transition: transform .28s ease, box-shadow .28s ease;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    border-color: rgba(255,255,255,0.12);
}}

/* profile circle */
.profile-circle {{
    width: 128px;
    height: 128px;
    border-radius: 50%;
    overflow: hidden;
    display:inline-block;
    border: 3px solid var(--accent);
    box-shadow: 0 8px 30px rgba(255,75,75,0.12);
}}

/* icons row */
.icon-row img {{
    width:48px;
    transition: transform .22s ease, filter .22s ease;
    filter: drop-shadow(0 6px 16px rgba(0,0,0,0.6));
    border-radius:8px;
}}
.icon-row img:hover {{
    transform: scale(1.15);
    filter: drop-shadow(0 12px 26px rgba(255,75,75,0.14));
}}

/* neon link + underline animation */
.link-underline {{
    color: #ffffff;
    text-decoration: none;
    position: relative;
    display: inline-block;
}}
.link-underline::after {{
    content: "";
    position: absolute;
    left: 0;
    bottom: -4px;
    height: 3px;
    width: 100%;
    background: linear-gradient(90deg, rgba(255,75,75,0.9), rgba(255,123,0,0.9));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform .28s cubic-bezier(.2,.9,.2,1);
    box-shadow: 0 6px 18px rgba(255,75,75,0.12);
}}
.link-underline:hover::after {{
    transform: scaleX(1);
}}

/* subtle animated glow on card links */
.card a {{
    color: #ffb3b3;
}}
.card a:hover {{
    text-shadow: 0 4px 18px rgba(255,75,75,0.18);
}}

/* particle canvas fill */
#particles-js {{
    position: fixed;
    width: 100%;
    height: 100%;
    z-index: 0;
    top: 0;
    left: 0;
    pointer-events: none;
    opacity: 0.55;
}}
/* ensure content is above particles */
.streamlit-container {{
    position: relative;
    z-index: 5;
}}
</style>

<!-- particles.js container -->
<div id="particles-js"></div>

<!-- load tsParticles via CDN -->
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2.11.1/tsparticles.bundle.min.js"></script>
<script>
tsParticles.load("particles-js", {{
  "fullScreen": {{ "enable": false }},
  "particles": {{
    "number": {{ "value": 60 }},
    "color": {{ "value": ["#ff4b4b", "#ff7b00", "#ffffff"] }},
    "shape": {{ "type": "circle" }},
    "opacity": {{ "value": 0.08 }},
    "size": {{ "value": {{ "min": 1, "max": 4 }} }},
    "move": {{ "enable": true, "speed": 0.6, "direction": "none", "outModes": "out" }},
    "links": {{ "enable": true, "distance": 120, "color": "#ff4b4b", "opacity": 0.06, "width": 1 }}
  }},
  "interactivity": {{
    "events": {{
      "onHover": {{ "enable": true, "mode": "grab" }},
      "onClick": {{ "enable": false }}
    }},
    "modes": {{
      "grab": {{ "distance": 140, "links": {{ "opacity": 0.12 }} }}
    }}
  }}
}});
</script>
"""

# render the CSS + particles (using components.html to allow full HTML + JS)
components.html(particles_and_style, height=10)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="streamlit-container">
      <div class="header">
          <h1 style="margin:0;">Pakistani Car Price Predictor</h1>
          <div style="opacity:0.95; margin-top:6px; font-size:15px;">Real-market car pricing • 2024–2025</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # spacer
st.markdown('<div class="streamlit-container">', unsafe_allow_html=True)

# -------------------- Main centered layout --------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    # profile picture (use provided processed image)
    img_tag = ""
    if PROFILE_PATH.exists():
        img_path_rel = PROFILE_PATH.as_posix()
        img_tag = f'<img src="file://{img_path_rel}" class="profile-circle">'
    else:
        # fallback placeholder
        img_tag = '<img src="https://i.imgur.com/3j8BPdc.png" class="profile-circle">'

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:18px;">
            {img_tag}
            <div style="height:10px;"></div>
            <div style="font-size:22px; font-weight:600; color:#ffffff; margin-top:6px;">
                Muhammad Haris Afridi
            </div>
            <div style="color:#cfcfcf; margin-top:4px;">AI Engineer & Deep Learning Enthusiast</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # icons row
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:12px;" class="card">
            <div style="display:flex; justify-content:center; gap:36px; align-items:center;">
                <a href="https://github.com/harisyar-ai" target="_blank" title="GitHub">
                    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
                </a>
                <a href="https://www.linkedin.com/in/muhammad-haris-afridi-051395379" target="_blank" title="LinkedIn">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # info card with glassmorphism + animated underline links
    st.markdown(
        f"""
        <div class="card" style="padding:18px; margin-top:8px;">
            <div style="padding:10px; font-size:17px; text-align:left; color:#f1f1f1;">
                <b>GitHub:</b><br>
                <a class="link-underline" href="https://github.com/harisyar-ai" target="_blank">github.com/harisyar-ai</a>
            </div>

            <div style="padding:10px; font-size:17px; text-align:left; color:#f1f1f1;">
                <b>LinkedIn:</b><br>
                <a class="link-underline" href="https://www.linkedin.com/in/muhammad-haris-afridi-051395379" target="_blank">linkedin.com/in/muhammad-haris-afridi</a>
            </div>

            <div style="padding:10px; font-size:17px; text-align:left; color:#f1f1f1;">
                <b>Gmail:</b><br>
                mharisyar.ai@gmail.com
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("---")

    st.markdown(
        """
        <div class="card" style="text-align:center; padding:18px;">
            Passionate about building real-world AI applications,<br>
            machine learning models, and solving complex data problems.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
