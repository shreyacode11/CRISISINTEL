import os
import json
import base64
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
import folium
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from utils.db_helper import get_db, init_db
from utils.auth_helper import login_required, admin_required, people_required

from ml_services.flood_predictor import predict_flood
from ml_services.earthquake_predictor import predict_earthquake
from ml_services.cyclone_predictor import predict_cyclone
from ml_services.gemini_classifier import classify_help
from ml_services import yolo_detector


from ml_services.telegram_notify import (
    send_admin, send_admin_photo,
    send_people, send_people_photo,
    broadcast_people, esc,
    notify_ml_prediction_to_admin,
    notify_ml_prediction_to_people,
)
import io
import telepot

import urllib.request
import urllib.parse
FLOOD_ALERT_CLASSES = {"very_high", "high", "moderate"}


def flood_is_alert(label: str) -> bool:
    """Return True if the flood label indicates a serious threat."""
    if label is None:
        return False
    normalized = str(label).strip().lower().replace(" ", "_")
    return normalized in FLOOD_ALERT_CLASSES

def esc(text):
    """Escape HTML special characters for Telegram HTML parse mode."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def _notify_red_request(req_id, message, help_type, city, lat, lon):
    from ml_services.map_renderer import render_folium_map_to_png
    from utils.timeutil import now_ist
    import folium

    # Coerce coords safely
    try:
        lat = float(lat) if lat not in (None, "", "None") else None
        lon = float(lon) if lon not in (None, "", "None") else None
    except (TypeError, ValueError):
        lat, lon = None, None

    ts = now_ist().strftime("%d-%b-%Y %I:%M:%S %p")

    # ---------- Admin text ----------
    admin_text = (
        f"🚨 <b>RED ALERT — Request #{req_id}</b>\n"
        f"<b>Time (IST):</b> {ts}\n"
        f"<b>Type:</b> {esc(help_type).upper()}\n"
        f"<b>City:</b> {esc(city) or 'Unknown'}\n"
        f"<b>Message:</b> {esc(message)}\n"
    )
    send_admin(admin_text)

    # ---------- People bot text (city-routed) ----------
    people_text = (
        f"🚨 <b>RED ALERT — Request #{req_id}</b>\n"
        f"<b>Time (IST):</b> {ts}\n"
        f"<b>Type:</b> {esc(help_type).upper()}\n"
        f"<b>Area:</b> {esc(city) or 'Unknown'}\n"
        f"<b>Message:</b> {esc(message)}\n\n"
        f"Rescue teams have been notified. Please stay safe."
    )
    send_people(city, people_text)   # ← city-routed

    # ---------- Map generation ----------
    if lat is None or lon is None:
        send_admin("⚠️ No coordinates for this request, map not generated.")
        return

    try:
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
        folium.Marker(
            [lat, lon], popup="Victim Location",
            tooltip="Emergency",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(m)

        shelters = [
            {"name": "Central Relief Camp",       "lat": lat + 0.012, "lon": lon + 0.008},
            {"name": "Govt. High School Shelter", "lat": lat - 0.010, "lon": lon + 0.015},
            {"name": "Community Hall Bunker",     "lat": lat + 0.008, "lon": lon - 0.011},
            {"name": "Red Cross Center",          "lat": lat - 0.014, "lon": lon - 0.006},
        ]
        for s in shelters:
            folium.Marker(
                [s["lat"], s["lon"]], popup=s["name"],
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)
            folium.PolyLine(
                [[lat, lon], [s["lat"], s["lon"]]],
                color="#0b3d91", weight=3, opacity=0.7,
            ).add_to(m)

        map_html = m._repr_html_()
    except Exception as e:
        send_admin(f"❌ Map generation failed for #{req_id}: {e}")
        return

    png_bytes = render_folium_map_to_png(map_html)

    caption = (
        f"📍 <b>Route map for Request #{req_id}</b>\n"
        f"<b>City:</b> {esc(city) or 'Unknown'}\n"
        f"<b>Type:</b> {esc(help_type).upper()}\n"
        f"<b>Coords:</b> {lat:.4f}, {lon:.4f}"
    )

    if png_bytes:
        send_admin_photo(png_bytes, caption=caption)
        send_people_photo(city, png_bytes, caption=caption)   # ← city-routed
    else:
        send_admin(caption + "\n(Map rendering unavailable)")


# Small cache so we don't geocode the same city repeatedly
_CITY_COORDS_CACHE = {
    "chennai":    (13.0827, 80.2707),
    "mumbai":     (19.0760, 72.8777),
    "delhi":      (28.6139, 77.2090),
    "kolkata":    (22.5726, 88.3639),
    "bangalore":  (12.9716, 77.5946),
    "bengaluru":  (12.9716, 77.5946),
    "hyderabad":  (17.3850, 78.4867),
    "pune":       (18.5204, 73.8567),
    "kochi":      (9.9312, 76.2673),
    "coimbatore": (11.0168, 76.9558),
    "madurai":    (9.9252, 78.1198),
    "vizag":      (17.6868, 83.2185),
    "visakhapatnam": (17.6868, 83.2185),
    "bhubaneswar": (20.2961, 85.8245),
    "guwahati":   (26.1445, 91.7362),
}


def _geocode_city(city):
    """Return (lat, lon) for a city name, using cache then Nominatim."""
    if not city:
        return None
    key = city.strip().lower()
    if key in _CITY_COORDS_CACHE:
        return _CITY_COORDS_CACHE[key]

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?"
            + urllib.parse.urlencode({"q": city, "format": "json", "limit": 1})
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "CrisisIntel/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            _CITY_COORDS_CACHE[key] = (lat, lon)
            return (lat, lon)
    except Exception as e:
        print("Geocode failed:", e)
    return None


app = Flask(__name__)
app.config.from_object(Config)

from utils.timeutil import to_ist, to_ist_short, to_ist_time_only, now_ist

app.jinja_env.filters["ist"] = to_ist
app.jinja_env.filters["ist_short"] = to_ist_short
app.jinja_env.filters["ist_time"] = to_ist_time_only

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
init_db()


# ----------------------------------------------------------------
# HOME
# ----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------
# AUTH
# ----------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        role = request.form["role"]
        city = request.form.get("city", "").strip()

        if role not in ("admin", "people"):
            flash("Invalid role.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (name,email,password,role,city) VALUES (?,?,?,?,?)",
                (name, email, generate_password_hash(password), role, city),
            )
            db.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration failed: {e}", "danger")
        finally:
            db.close()

    return render_template("auth/register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        db.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["role"] = user["role"]
            session["city"] = user["city"]
            flash(f"Welcome {user['name']}!", "success")
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("people_home"))

        flash("Invalid credentials.", "danger")

    return render_template("auth/login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))


# ================================================================
# ADMIN ROUTES
# ================================================================
@app.route("/admin/dashboard")
@login_required
@admin_required
def admin_dashboard():
    db = get_db()

    # Stats
    stats = {
        "total_requests": db.execute("SELECT COUNT(*) c FROM help_requests").fetchone()["c"],
        "pending": db.execute("SELECT COUNT(*) c FROM help_requests WHERE status='pending'").fetchone()["c"],
        "helped": db.execute("SELECT COUNT(*) c FROM help_requests WHERE status='helped'").fetchone()["c"],
        "food": db.execute("SELECT COUNT(*) c FROM help_requests WHERE help_type='food'").fetchone()["c"],
        "medical": db.execute("SELECT COUNT(*) c FROM help_requests WHERE help_type='medical'").fetchone()["c"],
        "shelter": db.execute("SELECT COUNT(*) c FROM help_requests WHERE help_type='shelter'").fetchone()["c"],
        "red": db.execute("SELECT COUNT(*) c FROM help_requests WHERE severity='red'").fetchone()["c"],
        "yellow": db.execute("SELECT COUNT(*) c FROM help_requests WHERE severity='yellow'").fetchone()["c"],
        "green": db.execute("SELECT COUNT(*) c FROM help_requests WHERE severity='green'").fetchone()["c"],
    }

    # Region-wise counts
    region_rows = db.execute(
        "SELECT city, COUNT(*) c FROM help_requests WHERE city IS NOT NULL GROUP BY city ORDER BY c DESC"
    ).fetchall()
    regions = [{"city": r["city"], "count": r["c"]} for r in region_rows]

    # Prediction history
    history = db.execute(
        "SELECT * FROM prediction_history ORDER BY id DESC LIMIT 20"
    ).fetchall()

    db.close()
    return render_template(
        "admin/dashboard.html",
        stats=stats, regions=regions, history=history
    )


# ---------- ML Predictions ----------
@app.route("/admin/predict/flood", methods=["GET", "POST"])
@login_required
@admin_required
def admin_predict_flood():
    result = None
    if request.method == "POST":
        try:
            # Extract city separately
            city = request.form.get("city", "").strip()

            # All non-city fields are model inputs
            inputs = {
                k: v for k, v in request.form.items()
                if k not in ("csrf_token", "city")
            }

            label, conf = predict_flood(inputs)
            is_alert = flood_is_alert(label)

            result = {"label": label, "confidence": conf, "alert": is_alert}

            db = get_db()
            db.execute(
                """INSERT INTO prediction_history
                   (model_type, input_data, prediction, confidence, alert)
                   VALUES (?, ?, ?, ?, ?)""",
                ("flood",
                 json.dumps({"city": city, "inputs": inputs}),
                 str(label), conf or 0.0, 1 if is_alert else 0)
            )
            db.commit()
            db.close()

            # ---- Telegram ----
            summary = ", ".join(f"{k}={v}" for k, v in list(inputs.items())[:5])
            conf_pct = (conf * 100) if conf is not None and conf <= 1 else conf

            # Admin always gets it
            notify_ml_prediction_to_admin(
                model_type="flood", prediction=label, confidence=conf_pct,
                is_alert=is_alert, input_summary=summary, city=city,
            )
            # People bot — only if alert, only for this city
            notify_ml_prediction_to_people(
                model_type="flood", prediction=label, confidence=conf_pct,
                is_alert=is_alert, city=city, input_summary=summary,
            )

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("admin/predict_flood.html", result=result)

@app.route("/admin/predict/earthquake", methods=["GET", "POST"])
@login_required
@admin_required
def admin_predict_earthquake():
    result = None
    if request.method == "POST":
        try:
            city = request.form.get("city", "").strip()
            mag = float(request.form["magnitude"])
            dep = float(request.form["depth"])
            cdi = float(request.form["cdi"])
            mmi = float(request.form["mmi"])
            sig = float(request.form["sig"])

            label, conf = predict_earthquake(mag, dep, cdi, mmi, sig)

            alert_classes = {"green", "low", "none", "no_alert", "safe"}
            is_alert = str(label).strip().lower() not in alert_classes

            result = {"label": label, "confidence": conf, "alert": is_alert}

            db = get_db()
            db.execute(
                """INSERT INTO prediction_history
                   (model_type, input_data, prediction, confidence, alert)
                   VALUES (?, ?, ?, ?, ?)""",
                ("earthquake",
                 json.dumps({"city": city, "magnitude": mag, "depth": dep,
                             "cdi": cdi, "mmi": mmi, "sig": sig}),
                 str(label), conf or 0.0, 1 if is_alert else 0)
            )
            db.commit()
            db.close()

            summary = f"mag={mag}, depth={dep}, cdi={cdi}, mmi={mmi}, sig={sig}"
            conf_pct = (conf * 100) if conf is not None and conf <= 1 else conf

            notify_ml_prediction_to_admin(
                model_type="earthquake", prediction=label, confidence=conf_pct,
                is_alert=is_alert, input_summary=summary, city=city,
            )
            notify_ml_prediction_to_people(
                model_type="earthquake", prediction=label, confidence=conf_pct,
                is_alert=is_alert, city=city, input_summary=summary,
            )

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("admin/predict_earthquake.html", result=result)

@app.route("/admin/predict/cyclone", methods=["GET", "POST"])
@login_required
@admin_required
def admin_predict_cyclone():
    result = None
    if request.method == "POST":
        try:
            city = request.form.get("city", "").strip()
            params = {
                "Sea_Surface_Temperature": float(request.form["sst"]),
                "Atmospheric_Pressure":    float(request.form["pressure"]),
                "Humidity":                float(request.form["humidity"]),
                "Wind_Shear":              float(request.form["wind_shear"]),
                "Vorticity":               float(request.form["vorticity"]),
                "Latitude":                float(request.form["latitude"]),
                "Ocean_Depth":             float(request.form["ocean_depth"]),
                "Proximity_to_Coastline":  float(request.form["proximity"]),
                "Pre_existing_Disturbance": int(request.form["disturbance"]),
            }
            label, conf = predict_cyclone(params)

            is_alert = (
                "cyclone" in str(label).lower()
                or str(label).lower() in ("yes", "1", "high")
            )

            result = {"label": label, "confidence": conf, "alert": is_alert}

            db = get_db()
            db.execute(
                """INSERT INTO prediction_history
                   (model_type, input_data, prediction, confidence, alert)
                   VALUES (?, ?, ?, ?, ?)""",
                ("cyclone", json.dumps({"city": city, **params}),
                 str(label), conf or 0.0, 1 if is_alert else 0)
            )
            db.commit()
            db.close()

            summary = (
                f"SST={params['Sea_Surface_Temperature']}, "
                f"Pressure={params['Atmospheric_Pressure']}, "
                f"Humidity={params['Humidity']}"
            )
            # Cyclone conf is already 0–100
            notify_ml_prediction_to_admin(
                model_type="cyclone", prediction=label, confidence=conf,
                is_alert=is_alert, input_summary=summary, city=city,
            )
            notify_ml_prediction_to_people(
                model_type="cyclone", prediction=label, confidence=conf,
                is_alert=is_alert, city=city, input_summary=summary,
            )

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("admin/predict_cyclone.html", result=result)

# ---------- Requests Management ----------
@app.route("/admin/requests")
@login_required
@admin_required
def admin_requests():
    filter_sev = request.args.get("severity", "")
    filter_type = request.args.get("type", "")
    filter_city = request.args.get("city", "")

    q = "SELECT h.*, u.name uname, u.email uemail FROM help_requests h JOIN users u ON h.user_id=u.id WHERE 1=1"
    params = []
    if filter_sev:
        q += " AND h.severity=?"; params.append(filter_sev)
    if filter_type:
        q += " AND h.help_type=?"; params.append(filter_type)
    if filter_city:
        q += " AND h.city=?"; params.append(filter_city)
    q += " ORDER BY h.id DESC"

    db = get_db()
    reqs = db.execute(q, params).fetchall()
    cities = db.execute("SELECT DISTINCT city FROM help_requests WHERE city IS NOT NULL").fetchall()
    db.close()

    return render_template(
        "admin/requests.html",
        requests=reqs, cities=[c["city"] for c in cities],
        filter_sev=filter_sev, filter_type=filter_type, filter_city=filter_city
    )


@app.route("/admin/request/<int:rid>/toggle", methods=["POST"])
@login_required
@admin_required
def admin_toggle_request(rid):
    db = get_db()
    row = db.execute(
        "SELECT h.*, u.name uname FROM help_requests h JOIN users u ON h.user_id=u.id WHERE h.id=?",
        (rid,)
    ).fetchone()

    if row:
        new_status = "helped" if row["status"] == "pending" else "pending"
        db.execute("UPDATE help_requests SET status=? WHERE id=?", (new_status, rid))
        db.commit()

        # ---------- Notify people when admin says "helped" ----------
        if new_status == "helped":
            from utils.timeutil import now_ist
            ts = now_ist().strftime("%d-%b-%Y %I:%M %p")
            ack_text = (
                f"✅ <b>Help is on the way!</b>\n\n"
                f"<b>Request ID:</b> #{row['id']}\n"
                f"<b>Time (IST):</b> {ts}\n"
                f"<b>Type:</b> {row['help_type'].upper()}\n"
                f"<b>City:</b> {row['city'] or 'Unknown'}\n\n"
                f"Our rescue team has been notified and is "
                f"<b>reaching you shortly</b>. Please stay calm and "
                f"remain in a safe location.\n\n"
                f"— National Disaster Response Team"
            )
            send_people(row["city"], ack_text)   # ← city-routed

    db.close()
    return redirect(request.referrer or url_for("admin_requests"))


# ---------- Emergency Notes ----------
@app.route("/admin/notes", methods=["GET", "POST"])
@login_required
@admin_required
def admin_notes():
    db = get_db()
    if request.method == "POST":
        title = request.form["title"]
        message = request.form["message"]
        severity = request.form.get("severity", "moderate")
        city = request.form.get("city", "")
        db.execute(
            "INSERT INTO emergency_notes (title,message,severity,city,created_by) VALUES (?,?,?,?,?)",
            (title, message, severity, city, session["user_id"])
        )
        db.commit()
        flash("Note posted.", "success")

    notes = db.execute(
        "SELECT * FROM emergency_notes ORDER BY id DESC"
    ).fetchall()
    db.close()
    return render_template("admin/emergency_notes.html", notes=notes)


@app.route("/admin/note/<int:nid>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_note(nid):
    db = get_db()
    db.execute("DELETE FROM emergency_notes WHERE id=?", (nid,))
    db.commit()
    db.close()
    flash("Note deleted.", "info")
    return redirect(url_for("admin_notes"))


# ---------- Drone (YOLO) ----------
@app.route("/admin/drone")
@login_required
@admin_required
def admin_drone():
    return render_template("admin/drone.html")


@app.route("/admin/drone/frame", methods=["POST"])
@login_required
@admin_required
def admin_drone_frame():
    data = request.get_json()
    img_data = data["image"].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    city = session.get("city", "") or request.json.get("city", "")
    detections, annotated = yolo_detector.detect_frame(
        frame, city=city, notify_admin=True, save_snapshot=True
    )

    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    # Persist injured detections to DB (existing behaviour)
    if detections:
        db = get_db()
        for d in detections:
            if d["label"].lower() == "injured":
                db.execute(
                    "INSERT INTO drone_detections (label, confidence, city) VALUES (?,?,?)",
                    (d["label"], d["confidence"], city)
                )
        db.commit()
        db.close()

    return jsonify({
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": detections
    })


# ================================================================
# PEOPLE ROUTES
# ================================================================
@app.route("/people/home")
@login_required
@people_required
def people_home():
    db = get_db()
    notes = db.execute(
        "SELECT * FROM emergency_notes ORDER BY id DESC LIMIT 10"
    ).fetchall()
    my_reqs = db.execute(
        "SELECT * FROM help_requests WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()
    db.close()
    return render_template("people/home.html", notes=notes, requests=my_reqs)


@app.route("/people/request", methods=["GET", "POST"])
@login_required
@people_required
def people_request():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        lat = request.form.get("latitude") or None
        lon = request.form.get("longitude") or None
        city = request.form.get("city", "").strip()

        # ---------- Classify ----------
        try:
            classified = classify_help(message)
        except Exception as e:
            flash(f"AI classification failed: {e}", "danger")
            classified = {"help_type": "food", "severity": "yellow",
                          "tag_color": "yellow", "reason": "fallback"}

        # ---------- Fallback coords from city ----------
        if not lat or not lon:
            fallback = _geocode_city(city)
            if fallback:
                lat, lon = fallback

        # ---------- Store request ----------
        db = get_db()
        cur = db.execute(
            """INSERT INTO help_requests
               (user_id,raw_message,help_type,severity,tag_color,city,latitude,longitude)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session["user_id"], message, classified["help_type"],
             classified["severity"], classified["tag_color"], city,
             float(lat) if lat else None,
             float(lon) if lon else None)
        )
        req_id = cur.lastrowid
        db.commit()
        db.close()

        result = {
            "id": req_id,
            "message": message,
            "help_type": classified["help_type"],
            "severity": classified["severity"],
            "tag_color": classified["tag_color"],
            "reason": classified.get("reason", ""),
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
            "city": city,
        }
        flash("Request submitted successfully.", "success")

        # ============================================================
        # TELEGRAM NOTIFICATIONS
        # ============================================================
        if classified["severity"] == "red":
            _notify_red_request(
                req_id=req_id,
                message=message,
                help_type=classified["help_type"],
                city=city,
                lat=float(lat) if lat else None,
                lon=float(lon) if lon else None,
            )

    return render_template("people/request_help.html", result=result)


@app.route("/people/map")
@login_required
@people_required
def people_map():
    lat = float(request.args.get("lat", 20.5937))
    lon = float(request.args.get("lon", 78.9629))
    city = request.args.get("city", "")

    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")

    # User marker
    folium.Marker(
        [lat, lon],
        popup="Your location",
        tooltip="You are here",
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(m)

    # Shelter / bunker markers
    shelters = [
        {"name": "Central Relief Camp", "lat": lat + 0.012, "lon": lon + 0.008},
        {"name": "Govt. High School Shelter", "lat": lat - 0.010, "lon": lon + 0.015},
        {"name": "Community Hall Bunker", "lat": lat + 0.008, "lon": lon - 0.011},
        {"name": "Red Cross Center", "lat": lat - 0.014, "lon": lon - 0.006},
        {"name": "Municipal Safe Zone", "lat": lat + 0.005, "lon": lon + 0.018},
    ]

    for s in shelters:
        folium.Marker(
            [s["lat"], s["lon"]],
            popup=s["name"],
            tooltip=s["name"],
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(m)

        folium.PolyLine(
            [[lat, lon], [s["lat"], s["lon"]]],
            color="#0b3d91",
            weight=3,
            opacity=0.7,
            tooltip=f"Route to {s['name']}",
        ).add_to(m)

    map_html = m._repr_html_()
    return render_template("people/map.html", map_html=map_html, city=city, lat=lat, lon=lon)

# ================================================================
# ADMIN — Prediction History Page
# ================================================================
@app.route("/admin/history")
@login_required
@admin_required
def admin_history():
    f_model = request.args.get("model", "")
    f_alert = request.args.get("alert", "")
    f_date  = request.args.get("date", "")

    query = "SELECT * FROM prediction_history WHERE 1=1"
    params = []

    if f_model:
        query += " AND model_type = ?"
        params.append(f_model)

    if f_alert in ("0", "1"):
        query += " AND alert = ?"
        params.append(int(f_alert))

    if f_date:
        query += " AND DATE(created_at) = ?"
        params.append(f_date)

    query += " ORDER BY id DESC"

    db = get_db()
    rows = db.execute(query, params).fetchall()

    # Pretty-print JSON inputs
    history = []
    for r in rows:
        try:
            pretty = json.dumps(json.loads(r["input_data"]), indent=2)
        except Exception:
            pretty = r["input_data"] or ""
        history.append({
            "id": r["id"],
            "model_type": r["model_type"],
            "input_data_pretty": pretty,
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "alert": r["alert"],
            "created_at": r["created_at"],
        })

    # Counts per model
    counts = {
        "flood": db.execute(
            "SELECT COUNT(*) c FROM prediction_history WHERE model_type='flood'"
        ).fetchone()["c"],
        "earthquake": db.execute(
            "SELECT COUNT(*) c FROM prediction_history WHERE model_type='earthquake'"
        ).fetchone()["c"],
        "cyclone": db.execute(
            "SELECT COUNT(*) c FROM prediction_history WHERE model_type='cyclone'"
        ).fetchone()["c"],
    }
    db.close()

    return render_template(
        "admin/history.html",
        history=history,
        counts=counts,
        f_model=f_model,
        f_alert=f_alert,
        f_date=f_date,
    )


# ================================================================
# PEOPLE — My Requests Page
# ================================================================

@app.route("/people/my-requests")
@login_required
@people_required
def people_my_requests():
    f_status = request.args.get("status", "")

    db = get_db()
    q = "SELECT * FROM help_requests WHERE user_id = ?"
    params = [session["user_id"]]

    if f_status in ("pending", "helped"):
        q += " AND status = ?"
        params.append(f_status)

    q += " ORDER BY id DESC"
    rows = db.execute(q, params).fetchall()

    # Stats
    stats = {
        "total": db.execute(
            "SELECT COUNT(*) c FROM help_requests WHERE user_id=?",
            (session["user_id"],)
        ).fetchone()["c"],
        "pending": db.execute(
            "SELECT COUNT(*) c FROM help_requests WHERE user_id=? AND status='pending'",
            (session["user_id"],)
        ).fetchone()["c"],
        "helped": db.execute(
            "SELECT COUNT(*) c FROM help_requests WHERE user_id=? AND status='helped'",
            (session["user_id"],)
        ).fetchone()["c"],
        "red": db.execute(
            "SELECT COUNT(*) c FROM help_requests WHERE user_id=? AND severity='red'",
            (session["user_id"],)
        ).fetchone()["c"],
    }
    db.close()

    return render_template(
        "people/my_requests.html",
        requests=rows,
        stats=stats,
        f_status=f_status,
    )

@app.route("/api/flood/fields")
@login_required
@admin_required
def api_flood_fields():
    import joblib
    pkg = joblib.load("flood.pkl")
    fields = []
    for f in pkg["feature_columns"]:
        if f in pkg["feature_encoders"]:
            fields.append({
                "name": f, "type": "categorical",
                "options": list(pkg["feature_encoders"][f].classes_),
            })
        else:
            fields.append({"name": f, "type": "numeric"})
    return jsonify({"fields": fields})



# ================================================================
# API
# ================================================================
@app.route("/api/admin/requests_geojson")
@login_required
@admin_required
def api_requests_geojson():
    db = get_db()
    rows = db.execute(
        "SELECT id,city,latitude,longitude,severity,help_type,status FROM help_requests WHERE latitude IS NOT NULL"
    ).fetchall()
    db.close()

    color_map = {"red": "#e74c3c", "yellow": "#f1c40f", "green": "#2ecc71"}
    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
            "properties": {
                "id": r["id"], "city": r["city"], "severity": r["severity"],
                "help_type": r["help_type"], "status": r["status"],
                "color": color_map.get(r["severity"], "#888"),
            },
        })
    return jsonify({"type": "FeatureCollection", "features": features})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)